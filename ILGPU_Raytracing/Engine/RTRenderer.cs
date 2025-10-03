// ==============================
// Engine/RTRenderer.cs  (single-pass: GBuffer -> Shade+GI with ReSTIR DI inside)
// rngLockNoise semantics:
//   - _rngLockNoise == 0  => locked noise (integrator gets 0; frame-invariant RNG)
//   - _rngLockNoise != 0  => animated noise (integrator gets a new Random int each frame)
//
// Light dir anim uses dt (frame-rate independent).
// Adds multi-bounce GI with Russian roulette; ReSTIR-DI only at first visible surface.
// Reservoirs are SoA. Integrator has 'spp' to reduce per-frame sizzle.
// This version matches kernels that do temporal reprojection + prev-frame spatial reuse IN-KERNEL.
// ==============================
using System;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Mathematics;

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
        private Camera _prevCamera;

        private readonly Framebuffer _framebuffer;
        private readonly GBuffer _gbuffer;

        private readonly Action<Index1D, GBufferParams> _primaryKernel;
        private readonly Action<Index1D, IntegratorParams, SpecializedValue<int>> _integratorKernel;

        private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _blitKernel;
        private readonly Action<Index1D, ArrayView<int>, int, int, ArrayView<int>, int, int> _bilinearUpsampleKernel;

        private float _renderScale = 0.67f;
        private bool _enableTAAU = true;

        private int _enableTemporalReuse = 1;
        private int _enableSpatialReuse = 1;
        private int _rngLockNoise = 1; // 0=locked (frame-invariant), nonzero=animated (fresh random seed per frame)
        private int _spp = 2;

        private readonly RTTaa _taa;

        private MemoryBuffer1D<int, Stride1D.Dense> _lowColor;
        private MemoryBuffer1D<float, Stride1D.Dense> _lowDepthScratch;
        private MemoryBuffer1D<int, Stride1D.Dense> _lowObjIdScratch;
        private MemoryBuffer1D<int, Stride1D.Dense> _lowCamIdScratch;

        // ---- Sun (directional light) animation state (dt-based) ----
        private float _sunAzimuth = 0.0f;
        private float _sunElevation = 0.9f;
        private float _sunSpeedRadPerSec = 0.0f;

        public RTRenderer(RTWindow window, int deviceIndex = 0)
        {
            _window = window ?? throw new ArgumentNullException(nameof(window));
            _context = Context.Create(builder => builder.Cuda().EnableAlgorithms().Math(MathMode.Fast32BitOnly).Optimize(OptimizationLevel.O2));
            _cuda = _context.CreateCudaAccelerator(deviceIndex);
            _stream = _cuda.DefaultStream as CudaStream;

            _sceneManager = new SceneManager(_cuda);
            _sceneManager.BuildDefaultScene();
            _sceneManager.Commit(RebuildPolicy.Auto);

            _cameraController = new FlyCameraController(_window);

            int w = Math.Max(1, _window.Size.X);
            int h = Math.Max(1, _window.Size.Y);
            _camera = Camera.CreateCamera(w, h, 60f);
            _camera.Translate(new Float3(1, 0, -4));
            _prevCamera = _camera;

            _framebuffer = new Framebuffer(_cuda, _stream);
            _gbuffer = new GBuffer(_cuda, _stream);

            _primaryKernel = _cuda.LoadAutoGroupedStreamKernel<Index1D, GBufferParams>(RTRay.PrimaryVisibilityKernel);
            _integratorKernel = _cuda.LoadAutoGroupedStreamKernel<Index1D, IntegratorParams, SpecializedValue<int>>(RTRay.PathTraceKernel);

            _blitKernel = _cuda.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(BlitKernel);
            _bilinearUpsampleKernel = _cuda.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int, ArrayView<int>, int, int>(BilinearUpsampleKernel);

            _taa = new RTTaa(_cuda, _stream);
        }

        public CudaAccelerator Accelerator => _cuda;

        public void UpdateCamera(float dtSeconds) => _cameraController.Update(ref _camera, dtSeconds);

        // Optional knobs for the sun animation:
        public void SetSunParams(float speedRadPerSec, float elevationRad)
        {
            _sunSpeedRadPerSec = speedRadPerSec;
            _sunElevation = elevationRad;
        }

        public void RenderDirectToPbo(CudaGlInteropIndexBuffer pbo, int width, int height, int frame, float dt)
        {
            if (pbo is null) throw new ArgumentNullException(nameof(pbo));

            int outW = Math.Max(1, width);
            int outH = Math.Max(1, height);
            int outLen = outW * outH;

            float scale = _renderScale;
            int inW = Math.Max(1, (int)XMath.Round(outW * scale));
            int inH = Math.Max(1, (int)XMath.Round(outH * scale));
            int inLen = inW * inH;

            _gbuffer.EnsureLength(inLen);
            _framebuffer.EnsureLength(inLen);
            EnsureLowResBuffers(inLen);

            // Bake derived camera data required by kernels (forward/right/up/fovYRadians/aspect)
            BakeCameraDerived(ref _camera, inW, inH);
            BakeCameraDerived(ref _prevCamera, inW, inH);

            Camera cam = _camera;

            _sceneManager.GetDeviceViews(
                out var tlasNodes, out var tlasInstanceIndices, out var instances,
                out var blasNodes, out var spherePrimIdx, out var spheres,
                out var triPrimIdx, out var meshPositions, out var meshTris,
                out var meshTexcoords, out var meshTriUVs, out var triMatIndex,
                out var materials, out var texels, out var texInfos);

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

            var gp = new GBufferParams { width = inW, height = inH, frame = frame, cam = cam, views = dv, gb = _gbuffer.GetGpu() };
            _primaryKernel(new Index1D(inLen), gp);

            var fbLow = new GpuFramebuffer
            {
                color = _lowColor.View,
                depth = _lowDepthScratch.View,
                objectId = _lowObjIdScratch.View,
                cameraId = _lowCamIdScratch.View
            };

            // Reservoir ping-pong buffers
            _framebuffer.GetReservoirPair(0, frame, out var resPrev, out var resCur);

            int temporalSeed = (_rngLockNoise == 0) ? 0 : Random.Shared.Next(int.MinValue, int.MaxValue);

            // Animate directional light (dt-based, frame-rate independent)
            float dtClamped = XMath.Clamp(dt, 0f, 0.1f);
            _sunAzimuth += _sunSpeedRadPerSec * dtClamped;
            const float TwoPi = 6.28318530717958647692f;
            if (_sunAzimuth >= TwoPi) _sunAzimuth -= TwoPi; else if (_sunAzimuth < 0f) _sunAzimuth += TwoPi;

            Float3 sunDir = Float3.Normalize(new Float3(
                XMath.Cos(_sunAzimuth) * XMath.Cos(_sunElevation),
                XMath.Sin(_sunElevation),
                XMath.Sin(_sunAzimuth) * XMath.Cos(_sunElevation)
            ));

            // Single-pass shading + ReSTIR DI inside the kernel
            var ip = new IntegratorParams
            {
                width = inW,
                height = inH,
                frame = frame,
                cam = cam,
                prevCam = _prevCamera, // IMPORTANT: used for temporal reprojection in-kernel
                views = dv,
                gb = _gbuffer.GetGpu(),
                fb = fbLow,
                dirLightDir = sunDir,
                dirLightRadiance = new Float3(10, 10, 10),
                skyTintTop = new Float3(0.5f, 0.7f, 1.0f),
                skyTintBottom = new Float3(1.0f, 1.0f, 1.0f),
                debugCamSeq = 0,
                resPrev = resPrev, // read-only
                resCur = resCur,   // write-only for this frame
                enableSpatialReuse = _enableSpatialReuse,
                enableTemporalReuse = _enableTemporalReuse,
                rngLockNoise = temporalSeed,
                spp = _spp
            };

            int maxDepth = 3; // GI depth (includes diffuse bounces)
            _integratorKernel(new Index1D(inLen), ip, SpecializedValue.New(maxDepth));

            // Present
            pbo.MapCuda(_stream);
            var pboView = pbo.GetCudaArrayView();

            if (_enableTAAU)
            {
                _taa.Ensure(outW, outH);
                _taa.ResolveUpsample(
                    pboView,
                    _lowColor.View,
                    _lowObjIdScratch.View,
                    inW, inH,
                    outW, outH,
                    frame,
                    _prevCamera,
                    _camera
                );
            }
            else
            {
                if (inW == outW && inH == outH)
                    _blitKernel(new Index1D(outLen), _lowColor.View, pboView);
                else
                    _bilinearUpsampleKernel(new Index1D(outLen), _lowColor.View, inW, inH, pboView, outW, outH);
            }

            _cuda.Synchronize();
            pbo.UnmapCuda(_stream);

            _prevCamera = _camera;
        }

        // Bake the derived camera quantities the kernels expect:
        // forward/right/up unit vectors, aspect, and vertical FOV in radians.
        private static void BakeCameraDerived(ref Camera c, int pixelW, int pixelH)
        {
            // Center ray direction
            Float3 center = c.lowerLeft + c.horizontal * 0.5f + c.vertical * 0.5f;
            Float3 forward = Normalize(center - c.origin);
            Float3 up = Normalize(c.vertical);
            Float3 right = Normalize(Cross(forward, up));

            float focusDist = Length(center - c.origin);
            float halfHeight = 0.5f * Length(c.vertical);
            float tanHalfFov = (focusDist > 1e-6f) ? (halfHeight / focusDist) : halfHeight;
            float fovY = 2f * XMath.Atan(tanHalfFov);
            float aspect = (Length(c.horizontal) > 1e-6f && Length(c.vertical) > 1e-6f)
                ? (Length(c.horizontal) / Length(c.vertical))
                : ((float)pixelW / (float)XMath.Max(1, pixelH));

            // These fields must exist in your Camera struct as per the kernel expectations:
            c.forward = forward;
            c.up = up;
            c.right = right;
            c.fovYRadians = fovY;
            c.aspect = aspect;
        }

        private void EnsureLowResBuffers(int inLen)
        {
            if (_lowColor == null || _lowColor.Length < inLen)
            {
                _lowColor?.Dispose();
                _lowDepthScratch?.Dispose();
                _lowObjIdScratch?.Dispose();
                _lowCamIdScratch?.Dispose();

                _lowColor = _cuda.Allocate1D<int>(inLen);
                _lowDepthScratch = _cuda.Allocate1D<float>(inLen);
                _lowObjIdScratch = _cuda.Allocate1D<int>(inLen);
                _lowCamIdScratch = _cuda.Allocate1D<int>(1);
            }
        }

        private static void BlitKernel(Index1D index, ArrayView<int> srcRGBA8, ArrayView<int> dstRGBA8)
        {
            if (index >= dstRGBA8.Length || index >= srcRGBA8.Length) return;
            dstRGBA8[index] = srcRGBA8[index];
        }

        private static void BilinearUpsampleKernel(Index1D index, ArrayView<int> srcRGBA8, int srcW, int srcH, ArrayView<int> dstRGBA8, int dstW, int dstH)
        {
            if (index >= dstRGBA8.Length) return;

            int x = index % dstW;
            int y = index / dstW;

            float u = ((x + 0.5f) * (float)srcW / (float)dstW) - 0.5f;
            float v = ((y + 0.5f) * (float)srcH / (float)dstH) - 0.5f;

            int x0 = XMath.Clamp((int)XMath.Floor(u), 0, srcW - 1);
            int y0 = XMath.Clamp((int)XMath.Floor(v), 0, srcH - 1);
            int x1 = XMath.Clamp(x0 + 1, 0, srcW - 1);
            int y1 = XMath.Clamp(y0 + 1, 0, srcH - 1);

            float tx = XMath.Clamp(u - x0, 0f, 1f);
            float ty = XMath.Clamp(v - y0, 0f, 1f);

            int i00 = y0 * srcW + x0;
            int i10 = y0 * srcW + x1;
            int i01 = y1 * srcW + x0;
            int i11 = y1 * srcW + x1;

            Float3 c00 = UnpackRGB(srcRGBA8[i00]);
            Float3 c10 = UnpackRGB(srcRGBA8[i10]);
            Float3 c01 = UnpackRGB(srcRGBA8[i01]);
            Float3 c11 = UnpackRGB(srcRGBA8[i11]);

            Float3 cx0 = c00 * (1f - tx) + c10 * tx;
            Float3 cx1 = c01 * (1f - tx) + c11 * tx;
            Float3 c = cx0 * (1f - ty) + cx1 * ty;

            dstRGBA8[index] = PackRGBA8(c);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 UnpackRGB(int rgba8)
        {
            float r = ((rgba8 >> 16) & 255) * (1f / 255f);
            float g = ((rgba8 >> 8) & 255) * (1f / 255f);
            float b = (rgba8 & 255) * (1f / 255f);
            return new Float3(r, g, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PackRGBA8(Float3 c)
        {
            int R = ToByte(c.X);
            int G = ToByte(c.Y);
            int B = ToByte(c.Z);
            return (255 << 24) | (R << 16) | (G << 8) | B;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToByte(float x)
        {
            float c = XMath.Min(1f, XMath.Max(0f, x));
            return (int)(255.99f * c);
        }

        public void Dispose()
        {
            try { _stream?.Synchronize(); } catch { }
            _lowColor?.Dispose();
            _lowDepthScratch?.Dispose();
            _lowObjIdScratch?.Dispose();
            _lowCamIdScratch?.Dispose();

            _taa?.Dispose();
            _gbuffer?.Dispose();
            if (_cameraController is IDisposable d) d.Dispose();
            _sceneManager?.Dispose();
            _framebuffer?.Dispose();
            _stream?.Dispose();
            _cuda?.Dispose();
            _context?.Dispose();
        }

        // --- tiny vector helpers (host-side) ---
        private static float Length(Float3 v) => XMath.Sqrt(v.X * v.X + v.Y * v.Y + v.Z * v.Z);
        private static Float3 Normalize(Float3 v)
        {
            float len2 = v.X * v.X + v.Y * v.Y + v.Z * v.Z;
            float inv = XMath.Rsqrt(XMath.Max(1e-20f, len2));
            return new Float3(v.X * inv, v.Y * inv, v.Z * inv);
        }
        private static Float3 Cross(Float3 a, Float3 b) =>
            new Float3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);
    }
}
