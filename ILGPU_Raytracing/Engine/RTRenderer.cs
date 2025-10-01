// ==============================
// File: Engine/RTRenderer.cs
// Replaces your existing RTRenderer.cs (kernel + all static RT helpers moved into RTRay class; renderer stays host-only)
// ==============================
using System;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace ILGPU_Raytracing.Engine
{
    public sealed class RTRenderer : IDisposable
    {
        private readonly RTWindow _window;
        private readonly Context _context;
        private readonly CudaAccelerator _cuda;
        private readonly CudaStream _stream;
        private readonly SceneManager _sceneManager;
        private readonly CameraController _cameraController;

        private Camera _camera;
        private readonly object _camLock = new();

        private readonly Framebuffer _framebuffer;

        // UPDATED: kernel delegate now takes SpecializedValue<int> budgets
        private readonly Action<
            Index1D,
            KernelParams,
            SpecializedValue<int>, // diffuse
            SpecializedValue<int>, // reflection
            SpecializedValue<int>, // refraction
            SpecializedValue<int>  // shadow on/off
        > _rayKernel;

        public RTRenderer(RTWindow window, int deviceIndex = 0)
        {
            _window = window ?? throw new ArgumentNullException(nameof(window));
            _context = Context.Create(builder => builder.Cuda().EnableAlgorithms());
            _cuda = _context.CreateCudaAccelerator(deviceIndex);
            _stream = _cuda.CreateStream() as CudaStream;

            _sceneManager = new SceneManager(_cuda);
            _sceneManager.BuildDefaultScene();
            _sceneManager.Commit(RebuildPolicy.Auto);

            _cameraController = new FlyCameraController(_window);

            int w = Math.Max(1, _window.Size.X);
            int h = Math.Max(1, _window.Size.Y);
            _camera = Camera.CreateCamera(w, h, 60f);

            _framebuffer = new Framebuffer(_cuda, _stream);

            // UPDATED: load kernel with the new signature
            _rayKernel =
                _cuda.LoadAutoGroupedStreamKernel<
                    Index1D,
                    KernelParams,
                    SpecializedValue<int>,
                    SpecializedValue<int>,
                    SpecializedValue<int>,
                    SpecializedValue<int>
                >(RTRay.RayTraceKernel);
        }

        public CudaAccelerator Accelerator { get { return _cuda; } }
        public CudaStream Stream { get { return _stream; } }
        public SceneManager SceneManager { get { return _sceneManager; } }
        public Scene Scene { get { return _sceneManager.Scene; } }

        public void UpdateCamera(float dtSeconds)
        {
            lock (_camLock)
            {
                _cameraController.Update(ref _camera, dtSeconds);
            }
        }

        public void RenderToDeviceBuffer(int slot, int width, int height, int frame)
        {
            int length = Math.Max(1, width * height);
            _framebuffer.EnsureLength(length);

            Camera cam;
            lock (_camLock) cam = _camera;

            _sceneManager.GetDeviceViews(
                out var tlasNodes,
                out var tlasInstanceIndices,
                out var instances,
                out var blasNodes,
                out var spherePrimIdx,
                out var spheres,
                out var triPrimIdx,
                out var meshPositions,
                out var meshTris,
                out var meshTexcoords,
                out var meshTriUVs,
                out var triMatIndex,
                out var materials,
                out var texels,
                out var texInfos);

            using (var binding = _cuda.BindScoped())
            {
                SceneDeviceViews dv = default;
                dv.tlasNodes = tlasNodes;
                dv.tlasInstanceIndices = tlasInstanceIndices;
                dv.instances = instances;
                dv.blasNodes = blasNodes;
                dv.spherePrimIdx = spherePrimIdx;
                dv.spheres = spheres;
                dv.triPrimIdx = triPrimIdx;
                dv.meshPositions = meshPositions;
                dv.meshTris = meshTris;
                dv.meshTexcoords = meshTexcoords;
                dv.meshTriUVs = meshTriUVs;
                dv.triMatIndex = triMatIndex;
                dv.materials = materials;
                dv.texels = texels;
                dv.texInfos = texInfos;

                KernelParams kp = default;
                kp.width = width;
                kp.height = height;
                kp.frame = frame;
                kp.cam = cam;
                kp.views = dv;

                // you can tweak these at runtime; the SpecializedValue<> enables DCE in the kernel
                kp.diffuseRays = 1;
                kp.reflectionRays = 1;
                kp.refractionRays = 2;
                kp.shadowRays = 1;

                kp.fb = _framebuffer.GetGpu(slot);

                _rayKernel(
                    new Index1D(length),
                    kp,
                    SpecializedValue.New(kp.diffuseRays),
                    SpecializedValue.New(kp.reflectionRays),
                    SpecializedValue.New(kp.refractionRays),
                    SpecializedValue.New(kp.shadowRays != 0 ? 1 : 0) // pass as on/off
                );

                _stream.Synchronize();
                binding.Recover();
            }
        }

        public void CopyDeviceToPbo(int slot, CudaGlInteropIndexBuffer pbo, int width, int height)
        {
            if (pbo == null) throw new ArgumentNullException(nameof(pbo));
            if (!pbo.IsValid()) throw new InvalidOperationException("Interop PBO is not valid.");
            _framebuffer.EnsureLength(Math.Max(1, width * height));
            _framebuffer.CopyColorToPbo(slot, pbo);
        }

        public void DownloadFramebufferToCpu(int slot)
        {
            _framebuffer.DownloadToCpu(slot);
        }

        public void Synchronize()
        {
            try { _stream?.Synchronize(); } catch { }
        }

        public void Dispose()
        {
            try { _stream?.Synchronize(); } catch { }
            _framebuffer?.Dispose();
            if (_cameraController is IDisposable d) d.Dispose();
            _sceneManager?.Dispose();
            _stream?.Dispose();
            _cuda?.Dispose();
            _context?.Dispose();
        }
    }
}