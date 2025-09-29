using System;
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

        // Renderer-owned camera; GL thread updates it via UpdateCamera.
        private Camera _camera;
        private readonly object _camLock = new();

        // Double device framebuffers (no GL interop here).
        private MemoryBuffer1D<int, Stride1D.Dense>[] _deviceFrames = Array.Empty<MemoryBuffer1D<int, Stride1D.Dense>>();
        private int _deviceFrameLength = 0; // width*height currently allocated

        // Kernel
        private readonly Action<
            Index1D, ArrayView<int>, int, int, int, Camera,
            ArrayView<TLASNode>, ArrayView<int>, ArrayView<InstanceRecord>,
            ArrayView<BLASNode>, ArrayView<int>, ArrayView<Sphere>,
            ArrayView<int>, ArrayView<Float3>, ArrayView<MeshTri>> _rayKernel;

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

            _rayKernel = _cuda.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, int, int, int, Camera,
                ArrayView<TLASNode>, ArrayView<int>, ArrayView<InstanceRecord>,
                ArrayView<BLASNode>, ArrayView<int>, ArrayView<Sphere>,
                ArrayView<int>, ArrayView<Float3>, ArrayView<MeshTri>>(RayTraceKernel);
        }

        public CudaAccelerator Accelerator => _cuda;
        public CudaStream Stream => _stream;
        public SceneManager SceneManager => _sceneManager;
        public Scene Scene => _sceneManager.Scene;

        // ---------------- Camera ----------------

        public void UpdateCamera(float dtSeconds)
        {
            lock (_camLock)
            {
                _cameraController.Update(ref _camera, dtSeconds);
            }
        }

        // ---------------- Rendering (worker thread) ----------------

        /// <summary>
        /// Render into the renderer-owned device framebuffer for the given slot (0/1).
        /// No GL interop here; this is safe to call from the worker thread.
        /// </summary>
        public void RenderToDeviceBuffer(int slot, int width, int height, int frame)
        {
            EnsureDeviceFrames(width * height);

            Camera cam;
            lock (_camLock) cam = _camera; // snapshot for this frame

            _sceneManager.GetDeviceViews(out var tlasNodes, out var tlasInstanceIndices, out var instances,
                                         out var blasNodes, out var spherePrimIdx, out var spheres,
                                         out var triPrimIdx, out var meshPositions, out var meshTris);

            using (var binding = _cuda.BindScoped())
            {
                var fb = _deviceFrames[slot].View;
                _rayKernel(new Index1D(width * height), fb, width, height, frame, cam,
                           tlasNodes, tlasInstanceIndices, instances,
                           blasNodes, spherePrimIdx, spheres, triPrimIdx, meshPositions, meshTris);
                _stream.Synchronize();
                binding.Recover();
            }
        }

        /// <summary>
        /// GL thread: map the PBO and copy from device framebuffer (slot) into it (device→device).
        /// Unmaps when done.
        /// </summary>
        public void CopyDeviceToPbo(int slot, CudaGlInteropIndexBuffer pbo, int width, int height)
        {
            if (pbo == null) throw new ArgumentNullException(nameof(pbo));
            if (!pbo.IsValid()) throw new InvalidOperationException("Interop PBO is not valid.");

            EnsureDeviceFrames(width * height);

            using (var binding = _cuda.BindScoped())
            {
                pbo.MapCuda(_stream);
                var dst = pbo.GetCudaArrayView();     // device view into mapped PBO
                var src = _deviceFrames[slot].View;   // device framebuffer we rendered into

                // Device→device copy on the same accelerator/stream.
                src.CopyTo(dst);
                _stream.Synchronize();

                pbo.UnmapCuda(_stream);
                binding.Recover();
            }
        }

        private void EnsureDeviceFrames(int length)
        {
            if (length == _deviceFrameLength && _deviceFrames.Length == 2 && _deviceFrames[0] != null && _deviceFrames[1] != null)
                return;

            // Dispose old
            if (_deviceFrames != null)
            {
                for (int i = 0; i < _deviceFrames.Length; i++)
                    _deviceFrames[i]?.Dispose();
            }

            _deviceFrameLength = length;
            _deviceFrames = new MemoryBuffer1D<int, Stride1D.Dense>[2];

            using (var binding = _cuda.BindScoped())
            {
                _deviceFrames[0] = _cuda.Allocate1D<int>(length);
                _deviceFrames[1] = _cuda.Allocate1D<int>(length);
                binding.Recover();
            }
        }

        // ---------------- Kernel & math ----------------

        private static void RayTraceKernel(
            Index1D index, ArrayView<int> framebuffer, int width, int height, int frame, Camera cam,
            ArrayView<TLASNode> tlasNodes, ArrayView<int> tlasInstanceIndices, ArrayView<InstanceRecord> instances,
            ArrayView<BLASNode> blasNodes, ArrayView<int> spherePrimIdx, ArrayView<Sphere> spheres,
            ArrayView<int> triPrimIdx, ArrayView<Float3> meshPositions, ArrayView<MeshTri> meshTris)
        {
            if (index >= framebuffer.Length) return;

            int x = index % width;
            int y = index / width;

            float u = ((float)x + 0.5f) / (float)XMath.Max(1, width);
            float v = ((float)y + 0.5f) / (float)XMath.Max(1, height);

            Ray wray = Ray.GenerateRay(cam, u, v);

            float closestT = 1e30f;
            Float3 bestNormal = default;
            Float3 bestAlbedo = new Float3(1f, 1f, 1f);

            int cur = 0;
            while (cur != -1)
            {
                TLASNode n = tlasNodes[cur];
                if (IntersectAABB(wray, n.boundsMin, n.boundsMax, 0.001f, closestT))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int instIndex = tlasInstanceIndices[i];
                            InstanceRecord inst = instances[instIndex];

                            Ray iray = TransformRay(inst.worldToObject, wray);
                            float scale = inst.uniformScale > 0f ? inst.uniformScale : 1f;

                            bool hit;
                            float tObjClosest; Float3 normalObj; Float3 albedo;
                            int blasStart = inst.blasRoot;
                            int blasEnd = blasStart + inst.blasNodeCount;

                            if (inst.type == BlasType.SphereSet)
                                hit = TraverseBLAS_Sphere(iray, blasNodes, blasStart, blasEnd, spherePrimIdx, spheres, out tObjClosest, out normalObj, out albedo);
                            else
                                hit = TraverseBLAS_Tri(iray, blasNodes, blasStart, blasEnd, triPrimIdx, meshPositions, meshTris, out tObjClosest, out normalObj, out albedo);

                            if (hit)
                            {
                                float tWorld = tObjClosest / scale;
                                if (tWorld < closestT)
                                {
                                    closestT = tWorld;
                                    bestNormal = Float3.Normalize(TransformVector(inst.objectToWorld, normalObj));
                                    bestAlbedo = albedo;
                                }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else
                    {
                        cur = n.left;
                    }
                }
                else
                {
                    cur = n.skipIndex;
                }
            }

            Float3 color;
            if (closestT < 1e29f)
            {
                Float3 lightDir = Float3.Normalize(new Float3(-0.4f, 1f, -0.2f));
                float ndotl = XMath.Max(0f, Float3.Dot(bestNormal, lightDir));
                color = bestAlbedo * (0.15f + 0.85f * ndotl);
            }
            else
            {
                float t = 0.5f * (wray.dir.Y + 1.0f);
                Float3 c1 = new Float3(1f, 1f, 1f);
                Float3 c2 = new Float3(0.5f, 0.7f, 1.0f);
                color = c1 * (1f - t) + c2 * t;
            }

            int R = ToByte(color.X);
            int G = ToByte(color.Y);
            int B = ToByte(color.Z);
            framebuffer[index] = (255 << 24) | (R << 16) | (G << 8) | B; // 0xAARRGGBB (little-endian = BGRA bytes)
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TraverseBLAS_Sphere(Ray rayObj, ArrayView<BLASNode> blasNodes, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Sphere> spheres, out float tClosest, out Float3 nObj, out Float3 albedo)
        {
            tClosest = 1e30f; nObj = default; albedo = new Float3(1f, 1f, 1f);
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tClosest))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int prim = primIndices[i];
                            float t; Float3 nn;
                            if (IntersectSphere(rayObj, spheres[prim], out t, out nn))
                            {
                                if (t > 0.001f && t < tClosest)
                                {
                                    tClosest = t; nObj = nn; albedo = spheres[prim].albedo;
                                }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return tClosest < 1e29f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TraverseBLAS_Tri(Ray rayObj, ArrayView<BLASNode> blasNodes, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Float3> positions, ArrayView<MeshTri> tris, out float tClosest, out Float3 nObj, out Float3 albedo)
        {
            tClosest = 1e30f; nObj = default; albedo = new Float3(0.85f, 0.85f, 0.85f);
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tClosest))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int triIndex = primIndices[i];
                            MeshTri tri = tris[triIndex];
                            Float3 v0 = positions[tri.i0];
                            Float3 v1 = positions[tri.i1];
                            Float3 v2 = positions[tri.i2];

                            float t; Float3 nn;
                            if (IntersectTriangleMT(rayObj, v0, v1, v2, out t, out nn))
                            {
                                if (t > 0.001f && t < tClosest) { tClosest = t; nObj = nn; }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return tClosest < 1e29f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IntersectTriangleMT(Ray ray, Float3 v0, Float3 v1, Float3 v2, out float t, out Float3 n)
        {
            Float3 e1 = v1 - v0;
            Float3 e2 = v2 - v0;
            Float3 p = Float3.Cross(ray.dir, e2);
            float det = Float3.Dot(e1, p);
            if (XMath.Abs(det) < 1e-8f) { t = 0f; n = default; return false; }
            float invDet = 1f / det;
            Float3 tv = ray.origin - v0;
            float u = Float3.Dot(tv, p) * invDet;
            if (u < 0f || u > 1f) { t = 0f; n = default; return false; }
            Float3 q = Float3.Cross(tv, e1);
            float v = Float3.Dot(ray.dir, q) * invDet;
            if (v < 0f || u + v > 1f) { t = 0f; n = default; return false; }
            t = Float3.Dot(e2, q) * invDet;
            if (t <= 0f) { n = default; return false; }
            n = Float3.Normalize(Float3.Cross(e1, e2));
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Ray TransformRay(Affine3x4 m, Ray w)
        {
            Float3 o = TransformPoint(m, w.origin);
            Float3 d = TransformVector(m, w.dir);
            Float3 inv = new Float3(
                1f / (d.X != 0f ? d.X : 1e-8f),
                1f / (d.Y != 0f ? d.Y : 1e-8f),
                1f / (d.Z != 0f ? d.Z : 1e-8f));
            return new Ray { origin = o, dir = d, invDir = inv };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TransformPoint(Affine3x4 m, Float3 p) =>
            new Float3(
                m.m00 * p.X + m.m01 * p.Y + m.m02 * p.Z + m.m03,
                m.m10 * p.X + m.m11 * p.Y + m.m12 * p.Z + m.m13,
                m.m20 * p.X + m.m21 * p.Y + m.m22 * p.Z + m.m23);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TransformVector(Affine3x4 m, Float3 v) =>
            new Float3(
                m.m00 * v.X + m.m01 * v.Y + m.m02 * v.Z,
                m.m10 * v.X + m.m11 * v.Y + m.m12 * v.Z,
                m.m20 * v.X + m.m21 * v.Y + m.m22 * v.Z);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IntersectSphere(Ray ray, Sphere s, out float t, out Float3 n)
        {
            Float3 oc = ray.origin - s.center;
            float a = Float3.Dot(ray.dir, ray.dir);
            float b = 2f * Float3.Dot(oc, ray.dir);
            float c = Float3.Dot(oc, oc) - s.radius * s.radius;
            float disc = b * b - 4f * a * c;
            if (disc < 0f) { t = 0f; n = default; return false; }
            float sqrtD = XMath.Sqrt(disc);
            float t0 = (-b - sqrtD) / (2f * a);
            float t1 = (-b + sqrtD) / (2f * a);
            t = t0;
            if (t < 0.001f)
            {
                t = t1;
                if (t < 0.001f) { n = default; return false; }
            }
            Float3 p = ray.origin + ray.dir * t;
            n = Float3.Normalize(p - s.center);
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IntersectAABB(Ray ray, Float3 bmin, Float3 bmax, float tMin, float tMax)
        {
            float t1 = (bmin.X - ray.origin.X) * ray.invDir.X;
            float t2 = (bmax.X - ray.origin.X) * ray.invDir.X;
            float tmin = XMath.Min(t1, t2);
            float tmax = XMath.Max(t1, t2);

            t1 = (bmin.Y - ray.origin.Y) * ray.invDir.Y;
            t2 = (bmax.Y - ray.origin.Y) * ray.invDir.Y;
            tmin = XMath.Max(tmin, XMath.Min(t1, t2));
            tmax = XMath.Min(tmax, XMath.Max(t1, t2));

            t1 = (bmin.Z - ray.origin.Z) * ray.invDir.Z;
            t2 = (bmax.Z - ray.origin.Z) * ray.invDir.Z;
            tmin = XMath.Max(tmin, XMath.Min(t1, t2));
            tmax = XMath.Min(tmax, XMath.Max(t1, t2));

            return tmax >= XMath.Max(tmin, tMin) && tmin <= tMax;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToByte(float x)
        {
            float c = XMath.Min(1f, XMath.Max(0f, x));
            return (int)(255.99f * c);
        }

        public void Synchronize()
        {
            try { _stream?.Synchronize(); } catch { }
        }

        public void Dispose()
        {
            try { _stream?.Synchronize(); } catch { }

            if (_deviceFrames != null)
                for (int i = 0; i < _deviceFrames.Length; i++)
                    _deviceFrames[i]?.Dispose();

            if (_cameraController is IDisposable d) d.Dispose();
            _sceneManager?.Dispose();
            _stream?.Dispose();
            _cuda?.Dispose();
            _context?.Dispose();
        }
    }
}
