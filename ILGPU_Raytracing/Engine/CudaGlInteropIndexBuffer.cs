using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL4;

namespace ILGPU_Raytracing.Engine
{
    public enum CudaGraphicsMapFlags : uint
    {
        None = 0,
        ReadOnly = 1,
        WriteDiscard = 2
    }

    public static class CudaGlInterop
    {
        [DllImport("nvcuda", EntryPoint = "cuGraphicsGLRegisterBuffer")]
        public static extern CudaError RegisterBuffer(out IntPtr resource, int buffer, uint flags);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsUnregisterResource")]
        public static extern CudaError UnregisterResource(IntPtr resource);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsMapResources")]
        public static extern CudaError MapResources(int count, IntPtr resources, IntPtr stream);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsUnmapResources")]
        public static extern CudaError UnmapResources(int count, IntPtr resources, IntPtr stream);

        [DllImport("nvcuda", EntryPoint = "cuGraphicsResourceGetMappedPointer_v2")]
        public static extern CudaError GetMappedPointer(out IntPtr devicePtr, out int size, IntPtr resource);
    }

    // GL PixelUnpackBuffer registered with CUDA; exposes a typed ArrayView<int> while mapped
    public sealed class CudaGlInteropIndexBuffer : MemoryBuffer
    {
        private IntPtr _cudaResource;
        public int glBufferHandle;
        private State _state;
        private readonly int _elementCount;

        public CudaGlInteropIndexBuffer(int elementCount, CudaAccelerator accelerator)
            : base(accelerator, elementCount * sizeof(int), sizeof(int))
        {
            _elementCount = elementCount;

            // Create GL PixelUnpackBuffer (used by glTexSubImage2D uploads)
            glBufferHandle = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, glBufferHandle);
            GL.BufferData(BufferTarget.PixelUnpackBuffer, elementCount * sizeof(int), IntPtr.Zero, BufferUsageHint.StreamDraw);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

            // Register with CUDA (WriteDiscard is fine; CUDA fully overwrites it each frame)
            CudaException.ThrowIfFailed(
                CudaGlInterop.RegisterBuffer(out _cudaResource, glBufferHandle, (uint)CudaGraphicsMapFlags.WriteDiscard));

            _state = State.AvailableForGl;
        }

        public unsafe void MapCuda(CudaStream stream)
        {
            if (_state != State.AvailableForGl) return;

            IntPtr* pRes = stackalloc IntPtr[1];
            pRes[0] = _cudaResource;

            CudaException.ThrowIfFailed(
                CudaGlInterop.MapResources(1, new IntPtr(pRes), stream?.StreamPtr ?? IntPtr.Zero));

            _state = State.MappedToCuda;
        }

        public ArrayView<int> GetCudaArrayView()
        {
            if (_state != State.MappedToCuda)
                throw new InvalidOperationException("PBO must be mapped to CUDA before accessing.");

            CudaException.ThrowIfFailed(
                CudaGlInterop.GetMappedPointer(out var devicePtr, out var byteSize, _cudaResource));

            Trace.Assert(byteSize == _elementCount * sizeof(int));
            NativePtr = devicePtr; // tell ILGPU where the memory lives

            return AsArrayView<int>(0, _elementCount);
        }

        public unsafe void UnmapCuda(CudaStream stream)
        {
            if (_state != State.MappedToCuda) return;

            IntPtr* pRes = stackalloc IntPtr[1];
            pRes[0] = _cudaResource;

            CudaException.ThrowIfFailed(
                CudaGlInterop.UnmapResources(1, new IntPtr(pRes), stream?.StreamPtr ?? IntPtr.Zero));

            NativePtr = IntPtr.Zero;
            _state = State.AvailableForGl;
        }

        public bool IsValid() => glBufferHandle != 0 && GL.IsBuffer(glBufferHandle);

        // ---- MemoryBuffer overrides (required) ----
        // We only allow these while the buffer is mapped to CUDA. No BindScoped() used (single stream).
        protected override void MemSet(AcceleratorStream stream, byte value, in ArrayView<byte> targetView)
        {
            if (_state != State.MappedToCuda) throw new InvalidOperationException("PBO must be mapped before MemSet.");
            if (stream is not CudaStream cs) throw new NotSupportedException("Only CUDA is supported.");

            CudaException.ThrowIfFailed(
                CudaAPI.CurrentAPI.Memset(
                    targetView.LoadEffectiveAddressAsPtr(),
                    value,
                    new IntPtr(targetView.LengthInBytes),
                    cs));
        }

        protected override void CopyTo(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            // Copy from this buffer -> targetView (device or host). Must be mapped.
            if (_state != State.MappedToCuda) throw new InvalidOperationException("PBO must be mapped before CopyTo.");
            if (stream is not CudaStream cs) throw new NotSupportedException("Only CUDA is supported.");

            var srcPtr = sourceView.LoadEffectiveAddressAsPtr();
            var dstPtr = targetView.LoadEffectiveAddressAsPtr();

            CudaException.ThrowIfFailed(
                CudaAPI.CurrentAPI.MemcpyAsync(
                    dstPtr,
                    srcPtr,
                    new IntPtr(targetView.LengthInBytes),
                    cs));
        }

        protected override void CopyFrom(AcceleratorStream stream, in ArrayView<byte> sourceView, in ArrayView<byte> targetView)
        {
            // Copy from sourceView -> this buffer. Must be mapped.
            if (_state != State.MappedToCuda) throw new InvalidOperationException("PBO must be mapped before CopyFrom.");
            if (stream is not CudaStream cs) throw new NotSupportedException("Only CUDA is supported.");

            var srcPtr = sourceView.LoadEffectiveAddressAsPtr();
            var dstPtr = targetView.LoadEffectiveAddressAsPtr();

            CudaException.ThrowIfFailed(
                CudaAPI.CurrentAPI.MemcpyAsync(
                    dstPtr,
                    srcPtr,
                    new IntPtr(targetView.LengthInBytes),
                    cs));
        }

        protected override unsafe void DisposeAcceleratorObject(bool disposing)
        {
            // Ensure unmapped and unregistered from CUDA
            try
            {
                if (_state == State.MappedToCuda)
                {
                    IntPtr* pRes = stackalloc IntPtr[1];
                    pRes[0] = _cudaResource;
                    CudaGlInterop.UnmapResources(1, new IntPtr(pRes), IntPtr.Zero);
                    _state = State.AvailableForGl;
                }
            }
            catch { /* best effort */ }

            try
            {
                if (_cudaResource != IntPtr.Zero)
                {
                    CudaGlInterop.UnregisterResource(_cudaResource);
                    _cudaResource = IntPtr.Zero;
                }
            }
            catch { /* best effort */ }

            if (disposing)
            {
                if (glBufferHandle != 0)
                {
                    GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);
                    GL.DeleteBuffer(glBufferHandle);
                    glBufferHandle = 0;
                }
            }

            base.Dispose();
        }

        private enum State { AvailableForGl, MappedToCuda }
    }
}
