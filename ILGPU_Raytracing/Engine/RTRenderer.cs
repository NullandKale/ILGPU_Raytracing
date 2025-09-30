// ==============================
// File: Engine/RTRenderer.cs
// Replaces your existing RTRenderer.cs (kernel now takes a single struct with all args)
// ==============================
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
    public struct SceneDeviceViews
    {
        public ArrayView<TLASNode> tlasNodes;
        public ArrayView<int> tlasInstanceIndices;
        public ArrayView<InstanceRecord> instances;
        public ArrayView<BLASNode> blasNodes;
        public ArrayView<int> spherePrimIdx;
        public ArrayView<Sphere> spheres;
        public ArrayView<int> triPrimIdx;
        public ArrayView<Float3> meshPositions;
        public ArrayView<MeshTri> meshTris;
        public ArrayView<Float2> meshTexcoords;
        public ArrayView<MeshTriUV> meshTriUVs;
        public ArrayView<int> triMatIndex;
        public ArrayView<MaterialRecord> materials;
        public ArrayView<RGBA32> texels;
        public ArrayView<TexInfo> texInfos;
    }

    public struct KernelParams
    {
        public ArrayView<int> framebuffer;
        public int width;
        public int height;
        public int frame;
        public Camera cam;
        public SceneDeviceViews views;
    }

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

        private MemoryBuffer1D<int, Stride1D.Dense>[] _deviceFrames = Array.Empty<MemoryBuffer1D<int, Stride1D.Dense>>();
        private int _deviceFrameLength = 0;

        private readonly Action<Index1D, KernelParams> _rayKernel;

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

            _rayKernel = _cuda.LoadAutoGroupedStreamKernel<Index1D, KernelParams>(RayTraceKernel);
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
            EnsureDeviceFrames(width * height);

            Camera cam;
            lock (_camLock) cam = _camera;

            _sceneManager.GetDeviceViews(out var tlasNodes, out var tlasInstanceIndices, out var instances, out var blasNodes, out var spherePrimIdx, out var spheres, out var triPrimIdx, out var meshPositions, out var meshTris, out var meshTexcoords, out var meshTriUVs, out var triMatIndex, out var materials, out var texels, out var texInfos);

            using (var binding = _cuda.BindScoped())
            {
                var fb = _deviceFrames[slot].View;

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
                kp.framebuffer = fb;
                kp.width = width;
                kp.height = height;
                kp.frame = frame;
                kp.cam = cam;
                kp.views = dv;

                _rayKernel(new Index1D(width * height), kp);
                _stream.Synchronize();
                binding.Recover();
            }
        }

        public void CopyDeviceToPbo(int slot, CudaGlInteropIndexBuffer pbo, int width, int height)
        {
            if (pbo == null) throw new ArgumentNullException(nameof(pbo));
            if (!pbo.IsValid()) throw new InvalidOperationException("Interop PBO is not valid.");

            EnsureDeviceFrames(width * height);

            using (var binding = _cuda.BindScoped())
            {
                pbo.MapCuda(_stream);
                var dst = pbo.GetCudaArrayView();
                var src = _deviceFrames[slot].View;
                src.CopyTo(dst);
                _stream.Synchronize();
                pbo.UnmapCuda(_stream);
                binding.Recover();
            }
        }

        private void EnsureDeviceFrames(int length)
        {
            if (length == _deviceFrameLength && _deviceFrames.Length == 2 && _deviceFrames[0] != null && _deviceFrames[1] != null) return;

            if (_deviceFrames != null)
            {
                for (int i = 0; i < _deviceFrames.Length; i++) _deviceFrames[i]?.Dispose();
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

        private static void RayTraceKernel(Index1D index, KernelParams k)
        {
            if (index >= k.framebuffer.Length) return;

            int x = index % k.width;
            int y = index / k.width;

            float u = ((float)x + 0.5f) / (float)XMath.Max(1, k.width);
            float v = ((float)y + 0.5f) / (float)XMath.Max(1, k.height);

            Ray wray = Ray.GenerateRay(k.cam, u, v);

            float closestT = 1e30f;
            Float3 bestNormal = default;
            Float3 bestAlbedo = new Float3(1f, 1f, 1f);

            int hitTri = -1;
            float hitBU = 0f;
            float hitBV = 0f;

            int cur = 0;
            while (cur != -1)
            {
                TLASNode n = k.views.tlasNodes[cur];
                if (IntersectAABB(wray, n.boundsMin, n.boundsMax, 0.001f, closestT))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int instIndex = k.views.tlasInstanceIndices[i];
                            InstanceRecord inst = k.views.instances[instIndex];

                            Ray iray = TransformRay(inst.worldToObject, wray);
                            float scale = inst.uniformScale > 0f ? inst.uniformScale : 1f;

                            bool hit;
                            float tObjClosest; Float3 normalObj; Float3 albedo; int triLocal; float bu; float bv;
                            int blasStart = inst.blasRoot;
                            int blasEnd = blasStart + inst.blasNodeCount;

                            if (inst.type == BlasType.SphereSet)
                            {
                                triLocal = -1; bu = 0f; bv = 0f;
                                hit = TraverseBLAS_Sphere(iray, k.views.blasNodes, blasStart, blasEnd, k.views.spherePrimIdx, k.views.spheres, out tObjClosest, out normalObj, out albedo);
                            }
                            else
                            {
                                hit = TraverseBLAS_Tri_Textured(iray, k.views.blasNodes, blasStart, blasEnd, k.views.triPrimIdx, k.views.meshPositions, k.views.meshTris, k.views.meshTexcoords, k.views.meshTriUVs, k.views.triMatIndex, k.views.materials, k.views.texels, k.views.texInfos, out tObjClosest, out normalObj, out albedo, out triLocal, out bu, out bv);
                            }

                            if (hit)
                            {
                                float tWorld = tObjClosest / scale;
                                if (tWorld < closestT)
                                {
                                    closestT = tWorld;
                                    bestNormal = Float3.Normalize(TransformVector(inst.objectToWorld, normalObj));
                                    bestAlbedo = albedo;
                                    hitTri = triLocal;
                                    hitBU = bu;
                                    hitBV = bv;
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
                float tbg = 0.5f * (wray.dir.Y + 1.0f);
                Float3 c1 = new Float3(1f, 1f, 1f);
                Float3 c2 = new Float3(0.5f, 0.7f, 1.0f);
                color = c1 * (1f - tbg) + c2 * tbg;
            }

            int R = ToByte(color.X);
            int G = ToByte(color.Y);
            int B = ToByte(color.Z);
            k.framebuffer[index] = (255 << 24) | (R << 16) | (G << 8) | B;
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
                                if (t > 0.001f && t < tClosest) { tClosest = t; nObj = nn; albedo = spheres[prim].albedo; }
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
        private static bool TraverseBLAS_Tri_Textured(Ray rayObj, ArrayView<BLASNode> blasNodes, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Float3> positions, ArrayView<MeshTri> tris, ArrayView<Float2> texcoords, ArrayView<MeshTriUV> triUVs, ArrayView<int> triMatIndex, ArrayView<MaterialRecord> materials, ArrayView<RGBA32> texels, ArrayView<TexInfo> texInfos, out float tClosest, out Float3 nObj, out Float3 albedo, out int triOut, out float buOut, out float bvOut)
        {
            tClosest = 1e30f; nObj = default; albedo = new Float3(0.85f, 0.85f, 0.85f); triOut = -1; buOut = 0f; bvOut = 0f;
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

                            float t; Float3 nn; float bu; float bv;
                            if (IntersectTriangleMT_Bary(rayObj, v0, v1, v2, out t, out nn, out bu, out bv))
                            {
                                if (t > 0.001f && t < tClosest)
                                {
                                    tClosest = t; nObj = nn; triOut = triIndex; buOut = bu; bvOut = bv;
                                    var tuv = triUVs[triIndex];
                                    Float2 t0 = texcoords[tuv.t0];
                                    Float2 t1 = texcoords[tuv.t1];
                                    Float2 t2 = texcoords[tuv.t2];
                                    float w = 1f - bu - bv;
                                    float uu = t0.X * w + t1.X * bu + t2.X * bv;
                                    float vv = t0.Y * w + t1.Y * bu + t2.Y * bv;
                                    int midx = triMatIndex[triIndex];
                                    MaterialRecord mat = materials[midx];
                                    if (mat.HasDiffuseMap != 0 && mat.DiffuseTexIndex >= 0 && mat.DiffuseTexIndex < texInfos.Length)
                                    {
                                        albedo = SampleTextureLinear(texels, texInfos[mat.DiffuseTexIndex], uu, vv);
                                    }
                                    else
                                    {
                                        albedo = mat.Kd;
                                    }
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
        private static bool IntersectTriangleMT_Bary(Ray ray, Float3 v0, Float3 v1, Float3 v2, out float t, out Float3 n, out float bu, out float bv)
        {
            Float3 e1 = v1 - v0;
            Float3 e2 = v2 - v0;
            Float3 p = Float3.Cross(ray.dir, e2);
            float det = Float3.Dot(e1, p);
            if (XMath.Abs(det) < 1e-8f) { t = 0f; n = default; bu = 0f; bv = 0f; return false; }
            float invDet = 1f / det;
            Float3 tv = ray.origin - v0;
            bu = Float3.Dot(tv, p) * invDet;
            if (bu < 0f || bu > 1f) { t = 0f; n = default; bv = 0f; return false; }
            Float3 q = Float3.Cross(tv, e1);
            bv = Float3.Dot(ray.dir, q) * invDet;
            if (bv < 0f || bu + bv > 1f) { t = 0f; n = default; return false; }
            t = Float3.Dot(e2, q) * invDet;
            if (t <= 0f) { n = default; return false; }
            n = Float3.Normalize(Float3.Cross(e1, e2));
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 SampleTextureLinear(ArrayView<RGBA32> texels, TexInfo info, float u, float v)
        {
            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));
            float x = fu * (float)(info.Width - 1);
            float y = fv * (float)(info.Height - 1);
            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(info.Width - 1, x0 + 1);
            int y1 = XMath.Min(info.Height - 1, y0 + 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;

            Float3 c00 = Texel(texels, info, x0, y0);
            Float3 c10 = Texel(texels, info, x1, y0);
            Float3 c01 = Texel(texels, info, x0, y1);
            Float3 c11 = Texel(texels, info, x1, y1);
            Float3 cx0 = c00 * (1f - tx) + c10 * tx;
            Float3 cx1 = c01 * (1f - tx) + c11 * tx;
            return cx0 * (1f - ty) + cx1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 Texel(ArrayView<RGBA32> texels, TexInfo info, int x, int y)
        {
            int idx = info.Offset + y * info.Width + x;
            RGBA32 p = texels[idx];
            return new Float3((float)p.R / 255f, (float)p.G / 255f, (float)p.B / 255f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Ray TransformRay(Affine3x4 m, Ray w)
        {
            Float3 o = TransformPoint(m, w.origin);
            Float3 d = TransformVector(m, w.dir);
            Float3 inv = new Float3(1f / (d.X != 0f ? d.X : 1e-8f), 1f / (d.Y != 0f ? d.Y : 1e-8f), 1f / (d.Z != 0f ? d.Z : 1e-8f));
            return new Ray { origin = o, dir = d, invDir = inv };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TransformPoint(Affine3x4 m, Float3 p)
        {
            return new Float3(m.m00 * p.X + m.m01 * p.Y + m.m02 * p.Z + m.m03, m.m10 * p.X + m.m11 * p.Y + m.m12 * p.Z + m.m13, m.m20 * p.X + m.m21 * p.Y + m.m22 * p.Z + m.m23);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Float3 TransformVector(Affine3x4 m, Float3 v)
        {
            return new Float3(m.m00 * v.X + m.m01 * v.Y + m.m02 * v.Z, m.m10 * v.X + m.m11 * v.Y + m.m12 * v.Z, m.m20 * v.X + m.m21 * v.Y + m.m22 * v.Z);
        }

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
            if (_deviceFrames != null) for (int i = 0; i < _deviceFrames.Length; i++) _deviceFrames[i]?.Dispose();
            if (_cameraController is IDisposable d) d.Dispose();
            _sceneManager?.Dispose();
            _stream?.Dispose();
            _cuda?.Dispose();
            _context?.Dispose();
        }
    }
}
