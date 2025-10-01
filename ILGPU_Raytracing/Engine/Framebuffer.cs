
// ==============================
// File: Engine/RTRenderer.cs
// Replaces your existing RTRenderer.cs (kernel + all static RT helpers moved into RTRay class; renderer stays host-only)
// ==============================
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ILGPU_Raytracing.Engine
{
    public struct GpuFramebuffer
    {
        public ArrayView<int> color;
        public ArrayView<float> depth;
        public ArrayView<int> objectId;
    }

    public sealed class Framebuffer : IDisposable
    {
        private readonly CudaAccelerator _cuda;
        private readonly CudaStream _stream;

        private MemoryBuffer1D<int, Stride1D.Dense>[] _color = new MemoryBuffer1D<int, Stride1D.Dense>[2];
        private MemoryBuffer1D<float, Stride1D.Dense>[] _depth = new MemoryBuffer1D<float, Stride1D.Dense>[2];
        private MemoryBuffer1D<int, Stride1D.Dense>[] _objectId = new MemoryBuffer1D<int, Stride1D.Dense>[2];

        private int[][] _cpuColor = new int[2][];
        private float[][] _cpuDepth = new float[2][];
        private int[][] _cpuObjectId = new int[2][];

        private int _length = 0;

        public Framebuffer(CudaAccelerator cuda, CudaStream stream) { _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda)); _stream = stream ?? throw new ArgumentNullException(nameof(stream)); }

        public int Length { get { return _length; } }

        public void EnsureLength(int length)
        {
            if (length <= 0) { throw new ArgumentOutOfRangeException(nameof(length)); }
            if (length == _length && _color[0] != null && _color[1] != null) return;

            DisposeBuffers();

            _length = length;
            using (var binding = _cuda.BindScoped())
            {
                for (int i = 0; i < 2; i++)
                {
                    _color[i] = _cuda.Allocate1D<int>(length);
                    _depth[i] = _cuda.Allocate1D<float>(length);
                    _objectId[i] = _cuda.Allocate1D<int>(length);
                    _cpuColor[i] = new int[length];
                    _cpuDepth[i] = new float[length];
                    _cpuObjectId[i] = new int[length];
                }
                binding.Recover();
            }
        }

        public GpuFramebuffer GetGpu(int slot)
        {
            if (slot != 0 && slot != 1) { throw new ArgumentOutOfRangeException(nameof(slot)); }
            if (_color[slot] == null) { throw new InvalidOperationException("Framebuffer not allocated. Call EnsureLength() first."); }
            GpuFramebuffer fb = default;
            fb.color = _color[slot].View;
            fb.depth = _depth[slot].View;
            fb.objectId = _objectId[slot].View;
            return fb;
        }

        public void DownloadToCpu(int slot)
        {
            if (slot != 0 && slot != 1) { throw new ArgumentOutOfRangeException(nameof(slot)); }
            if (_color[slot] == null) { throw new InvalidOperationException("Framebuffer not allocated."); }
            using (var binding = _cuda.BindScoped())
            {
                _color[slot].CopyToCPU(_cpuColor[slot]);
                _depth[slot].CopyToCPU(_cpuDepth[slot]);
                _objectId[slot].CopyToCPU(_cpuObjectId[slot]);
                _stream.Synchronize();
                binding.Recover();
            }
        }

        public int[] CpuColor(int slot)
        {
            if (slot != 0 && slot != 1) { throw new ArgumentOutOfRangeException(nameof(slot)); }
            return _cpuColor[slot];
        }

        public float[] CpuDepth(int slot)
        {
            if (slot != 0 && slot != 1) { throw new ArgumentOutOfRangeException(nameof(slot)); }
            return _cpuDepth[slot];
        }

        public int[] CpuObjectId(int slot)
        {
            if (slot != 0 && slot != 1) { throw new ArgumentOutOfRangeException(nameof(slot)); }
            return _cpuObjectId[slot];
        }

        public void CopyColorToPbo(int slot, CudaGlInteropIndexBuffer pbo)
        {
            if (slot != 0 && slot != 1) { throw new ArgumentOutOfRangeException(nameof(slot)); }
            if (pbo == null) { throw new ArgumentNullException(nameof(pbo)); }
            if (_color[slot] == null) { throw new InvalidOperationException("Framebuffer not allocated."); }
            using (var binding = _cuda.BindScoped())
            {
                pbo.MapCuda(_stream);
                var dst = pbo.GetCudaArrayView();
                var src = _color[slot].View;
                src.CopyTo(dst);
                _stream.Synchronize();
                pbo.UnmapCuda(_stream);
                binding.Recover();
            }
        }

        private void DisposeBuffers()
        {
            for (int i = 0; i < 2; i++)
            {
                _color[i]?.Dispose();
                _depth[i]?.Dispose();
                _objectId[i]?.Dispose();
                _color[i] = null;
                _depth[i] = null;
                _objectId[i] = null;
                _cpuColor[i] = null;
                _cpuDepth[i] = null;
                _cpuObjectId[i] = null;
            }
        }

        public void Dispose()
        {
            DisposeBuffers();
        }
    }
}