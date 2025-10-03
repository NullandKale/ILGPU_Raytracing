// ==============================
// GPU-side data and kernels (struct-oriented API) + Back-compat wrapper + ReSTIR DI (fixed reservoir reweighting)
// ==============================
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;

namespace ILGPU_Raytracing.Engine
{
    // ------------ Device views & scene services ------------
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TraceClosest(Ray wray, out float closestT, out Float3 bestNormal, out Float3 bestAlbedo, out int bestObjId, out int bestShade, out float bestIor)
        {
            closestT = 1e30f; bestNormal = default; bestAlbedo = new Float3(1f, 1f, 1f); bestObjId = -1; bestShade = 0; bestIor = 1f;
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

                            float tObjClosest; Float3 normalObj; Float3 albedo; int triLocal; float bu; float bv; int shade; float ior;
                            bool hit;
                            int blasStart = inst.blasRoot;
                            int blasEnd = blasStart + inst.blasNodeCount;

                            if (inst.type == BlasType.SphereSet)
                            {
                                triLocal = -1; bu = 0f; bv = 0f; shade = 0; ior = 1f;
                                hit = TraverseBLAS_Sphere(iray, blasStart, blasEnd, spherePrimIdx, spheres, out tObjClosest, out normalObj, out albedo, out shade, out ior);
                            }
                            else
                            {
                                shade = 0; ior = 1f;
                                hit = TraverseBLAS_Tri_Textured(iray, blasStart, blasEnd, triPrimIdx, meshPositions, meshTris, meshTexcoords, meshTriUVs, triMatIndex, materials, out tObjClosest, out normalObj, out albedo, out triLocal, out bu, out bv);
                            }

                            if (hit)
                            {
                                float tWorld = tObjClosest / scale;
                                if (tWorld < closestT)
                                {
                                    closestT = tWorld;
                                    bestNormal = Float3.Normalize(TransformVector(inst.objectToWorld, normalObj));
                                    bestAlbedo = albedo;
                                    bestObjId = triLocal;
                                    bestShade = shade;
                                    bestIor = ior;
                                }
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return closestT < 1e29f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool ShadowOcclusion(Ray srayWorld, float tMaxWorld)
        {
            int cur = 0;
            while (cur != -1)
            {
                TLASNode n = tlasNodes[cur];
                if (IntersectAABB(srayWorld, n.boundsMin, n.boundsMax, 0.001f, tMaxWorld))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int instIndex = tlasInstanceIndices[i];
                            InstanceRecord inst = instances[instIndex];

                            Ray srayObj = TransformRay(inst.worldToObject, srayWorld);
                            float scale = inst.uniformScale > 0f ? inst.uniformScale : 1f;
                            float tMaxObj = tMaxWorld * scale;

                            bool blocked = (inst.type == BlasType.SphereSet)
                                ? AnyHit_Sphere(srayObj, inst.blasRoot, inst.blasRoot + inst.blasNodeCount, spherePrimIdx, spheres, tMaxObj)
                                : AnyHit_Tri_Textured(srayObj, inst, inst.blasRoot, inst.blasRoot + inst.blasNodeCount, triPrimIdx, meshPositions, meshTris, meshTexcoords, meshTriUVs, triMatIndex, materials, tMaxObj);
                            if (blocked) return true;
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TraverseBLAS_Sphere(Ray rayObj, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Sphere> spheres, out float tClosest, out Float3 nObj, out Float3 albedo, out int shading, out float ior)
        {
            tClosest = 1e30f; nObj = default; albedo = new Float3(1f, 1f, 1f); shading = 0; ior = 1f;
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
                                    tClosest = t;
                                    nObj = nn;
                                    var s = spheres[prim];
                                    Float3 kd = s.material.Kd;
                                    Float3 col = (kd.X == 0f && kd.Y == 0f && kd.Z == 0f) ? s.albedo : kd;
                                    if (s.material.HasDiffuseMap != 0 && s.material.DiffuseTexIndex >= 0 && s.material.DiffuseTexIndex < texInfos.Length)
                                    {
                                        const float PI = 3.14159265358979323846f;
                                        float u = 0.5f + XMath.Atan2(nn.Z, nn.X) / (2f * PI);
                                        float v = XMath.Acos(XMath.Min(1f, XMath.Max(-1f, nn.Y))) / PI;
                                        float aTmp;
                                        col = SampleTextureLinearRGB_A(texInfos[s.material.DiffuseTexIndex], u, v, out aTmp);
                                    }
                                    albedo = col;
                                    shading = s.shading;
                                    ior = s.ior > 0f ? s.ior : 1f;
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
        private bool TraverseBLAS_Tri_Textured(Ray rayObj, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Float3> positions, ArrayView<MeshTri> tris, ArrayView<Float2> texcoords, ArrayView<MeshTriUV> triUVs, ArrayView<int> triMatIndex, ArrayView<MaterialRecord> materials, out float tClosest, out Float3 nObj, out Float3 albedo, out int triOut, out float buOut, out float bvOut)
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
                                int midx = triMatIndex[triIndex];
                                MaterialRecord mat = materials[midx];

                                if (t > 0.001f && t < tClosest)
                                {
                                    var tuv = triUVs[triIndex];
                                    Float2 t0 = texcoords[tuv.t0];
                                    Float2 t1 = texcoords[tuv.t1];
                                    Float2 t2 = texcoords[tuv.t2];
                                    float w = 1f - bu - bv;
                                    float uu = t0.X * w + t1.X * bu + t2.X * bv;
                                    float vv = t0.Y * w + t1.Y * bu + t2.Y * bv;

                                    float alpha = 1f;
                                    Float3 kdCol = mat.Kd;

                                    if (mat.HasDiffuseMap != 0 && mat.DiffuseTexIndex >= 0 && mat.DiffuseTexIndex < texInfos.Length)
                                        kdCol = SampleTextureLinear(texInfos[mat.DiffuseTexIndex], uu, vv);

                                    if (mat.HasAlphaMap != 0 && mat.AlphaTexIndex >= 0 && mat.AlphaTexIndex < texInfos.Length)
                                        alpha = SampleMaskLinear(texInfos[mat.AlphaTexIndex], uu, vv);

                                    if (alpha < mat.AlphaCutoff) { continue; }

                                    tClosest = t;
                                    nObj = nn;
                                    if (mat.TwoSided != 0 && Float3.Dot(nObj, rayObj.dir) > 0f) nObj = nObj * -1f;
                                    albedo = kdCol;
                                    triOut = triIndex;
                                    buOut = bu;
                                    bvOut = bv;
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
        private bool AnyHit_Sphere(Ray rayObj, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Sphere> spheres, float tMaxObj)
        {
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tMaxObj))
                {
                    if (n.count > 0)
                    {
                        int end = n.first + n.count;
                        for (int i = n.first; i < end; i++)
                        {
                            int prim = primIndices[i];
                            float t; Float3 _n;
                            if (IntersectSphere(rayObj, spheres[prim], out t, out _n))
                            {
                                if (t > 0.001f && t < tMaxObj) return true;
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool AnyHit_Tri_Textured(Ray rayObj, InstanceRecord inst, int blasStart, int blasEnd, ArrayView<int> primIndices, ArrayView<Float3> positions, ArrayView<MeshTri> tris, ArrayView<Float2> texcoords, ArrayView<MeshTriUV> triUVs, ArrayView<int> triMatIndex, ArrayView<MaterialRecord> materials, float tMaxObj)
        {
            int cur = blasStart;
            while (cur != -1 && cur < blasEnd)
            {
                BLASNode n = blasNodes[cur];
                if (IntersectAABB(rayObj, n.boundsMin, n.boundsMax, 0.001f, tMaxObj))
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
                                if (t <= 0.001f || t >= tMaxObj) continue;

                                int midx = triMatIndex[triIndex];
                                MaterialRecord mat = materials[midx];

                                if (mat.HasAlphaMap != 0 && mat.AlphaTexIndex >= 0 && mat.AlphaTexIndex < texInfos.Length)
                                {
                                    var tuv = triUVs[triIndex];
                                    Float2 t0 = texcoords[tuv.t0];
                                    Float2 t1 = texcoords[tuv.t1];
                                    Float2 t2 = texcoords[tuv.t2];
                                    float w = 1f - bu - bv;
                                    float uu = t0.X * w + t1.X * bu + t2.X * bv;
                                    float vv = t0.Y * w + t1.Y * bu + t2.Y * bv;

                                    float aPoint = SampleMaskPoint(texInfos[mat.AlphaTexIndex], uu, vv);
                                    float cutoff = mat.AlphaCutoff;
                                    const float Band = 0.10f;
                                    if (aPoint < cutoff - Band) { continue; }
                                    if (aPoint >= cutoff + Band) { return true; }

                                    float aLin = SampleMaskLinear(texInfos[mat.AlphaTexIndex], uu, vv);
                                    if (aLin < cutoff) { continue; }
                                }

                                return true;
                            }
                        }
                        cur = n.skipIndex;
                    }
                    else cur = n.left;
                }
                else cur = n.skipIndex;
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private RGBA32 TexelRaw(TexInfo info, int x, int y)
        {
            int w = info.Width;
            int h = info.Height;
            if (w <= 0 || h <= 0) return default;
            int sx = XMath.Max(0, XMath.Min(w - 1, x));
            int sy = XMath.Max(0, XMath.Min(h - 1, y));
            int idx = info.Offset + sy * w + sx;
            return texels[idx];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Luma01(RGBA32 p)
        {
            float r = p.R * (1f / 255f);
            float g = p.G * (1f / 255f);
            float b = p.B * (1f / 255f);
            return 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Float3 TexelRGB(TexInfo info, int x, int y)
        {
            var p = TexelRaw(info, x, y);
            return new Float3(p.R * (1f / 255f), p.G * (1f / 255f), p.B * (1f / 255f));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Float3 SampleTextureLinear(TexInfo info, float u, float v)
        {
            int w = info.Width, h = info.Height;
            if (w <= 0 || h <= 0) return new Float3(1f, 1f, 1f);

            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));

            float x = fu * (w - 1);
            float y = fv * (h - 1);

            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(w - 1, x0 + 1);
            int y1 = XMath.Min(h - 1, y0 + 1);

            float tx = x - x0;
            float ty = y - y0;

            Float3 c00 = TexelRGB(info, x0, y0);
            Float3 c10 = TexelRGB(info, x1, y0);
            Float3 c01 = TexelRGB(info, x0, y1);
            Float3 c11 = TexelRGB(info, x1, y1);

            Float3 cx0 = c00 * (1f - tx) + c10 * tx;
            Float3 cx1 = c01 * (1f - tx) + c11 * tx;
            return cx0 * (1f - ty) + cx1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float SampleMaskLinear(TexInfo info, float u, float v)
        {
            int w = info.Width, h = info.Height;
            if (w <= 0 || h <= 0) return 1f;

            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));

            float x = fu * (w - 1);
            float y = fv * (h - 1);

            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(w - 1, x0 + 1);
            int y1 = XMath.Min(h - 1, y0 + 1);

            float tx = x - x0;
            float ty = y - y0;

            float a00 = Luma01(TexelRaw(info, x0, y0));
            float a10 = Luma01(TexelRaw(info, x1, y0));
            float a01 = Luma01(TexelRaw(info, x0, y1));
            float a11 = Luma01(TexelRaw(info, x1, y1));

            float ax0 = a00 * (1f - tx) + a10 * tx;
            float ax1 = a01 * (1f - tx) + a11 * tx;
            return ax0 * (1f - ty) + ax1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float SampleMaskPoint(TexInfo info, float u, float v)
        {
            int w = info.Width, h = info.Height;
            if (w <= 0 || h <= 0) return 1f;

            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));
            int x = (int)XMath.Round(fu * (w - 1));
            int y = (int)XMath.Round(fv * (h - 1));
            return Luma01(TexelRaw(info, x, y));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Float3 SampleTextureLinearRGB_A(TexInfo info, float u, float v, out float a)
        {
            int w = info.Width, h = info.Height;
            if (w <= 0 || h <= 0) { a = 1f; return new Float3(1f, 1f, 1f); }

            float fu = u - XMath.Floor(u);
            float fv = 1f - (v - XMath.Floor(v));

            float x = fu * (w - 1);
            float y = fv * (h - 1);

            int x0 = (int)XMath.Floor(x);
            int y0 = (int)XMath.Floor(y);
            int x1 = XMath.Min(w - 1, x0 + 1);
            int y1 = XMath.Min(h - 1, y0 + 1);

            float tx = x - x0;
            float ty = y - y0;

            RGBA32 p00 = TexelRaw(info, x0, y0);
            RGBA32 p10 = TexelRaw(info, x1, y0);
            RGBA32 p01 = TexelRaw(info, x0, y1);
            RGBA32 p11 = TexelRaw(info, x1, y1);

            Float3 c00 = new Float3(p00.R * (1f / 255f), p00.G * (1f / 255f), p00.B * (1f / 255f));
            Float3 c10 = new Float3(p10.R * (1f / 255f), p10.G * (1f / 255f), p10.B * (1f / 255f));
            Float3 c01 = new Float3(p01.R * (1f / 255f), p01.G * (1f / 255f), p01.B * (1f / 255f));
            Float3 c11 = new Float3(p11.R * (1f / 255f), p11.G * (1f / 255f), p11.B * (1f / 255f));

            float a00 = p00.A * (1f / 255f);
            float a10 = p10.A * (1f / 255f);
            float a01 = p01.A * (1f / 255f);
            float a11 = p11.A * (1f / 255f);

            Float3 cx0 = c00 * (1f - tx) + c10 * tx;
            Float3 cx1 = c01 * (1f - tx) + c11 * tx;
            float ax0 = a00 * (1f - tx) + a10 * tx;
            float ax1 = a01 * (1f - tx) + a11 * tx;

            a = ax0 * (1f - ty) + ax1 * ty;
            return cx0 * (1f - ty) + cx1 * ty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Ray TransformRay(Affine3x4 m, Ray w)
        {
            Float3 o = TransformPoint(m, w.origin);
            Float3 d = TransformVector(m, w.dir);
            Float3 inv = RTRay.InvDir(d);
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
    }
}
