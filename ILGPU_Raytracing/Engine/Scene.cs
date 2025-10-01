// ==============================
// File: Engine/Scene.cs
// Replaces your existing Scene.cs (adds UVs/materials/textures pipeline and uploads + alpha map remap/upload)
// ==============================
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ILGPU_Raytracing.Engine
{
    public sealed class Scene : IDisposable
    {
        private readonly CudaAccelerator _cuda;

        private TLASNode[] _hTLASNodes = Array.Empty<TLASNode>();
        private int[] _hTLASInstanceIndices = Array.Empty<int>();
        private InstanceRecord[] _hInstances = Array.Empty<InstanceRecord>();

        private readonly List<BLASNode> _hBLASNodes = new List<BLASNode>();

        private readonly List<int> _hSpherePrimIndices = new List<int>();
        private readonly List<Sphere> _hSpheres = new List<Sphere>();

        private readonly List<int> _hTriPrimIndices = new List<int>();
        private readonly List<Float3> _hMeshPositions = new List<Float3>();
        private readonly List<MeshTri> _hMeshTris = new List<MeshTri>();

        // UVs/materials/textures
        private readonly List<Float2> _hMeshTexcoords = new List<Float2>();
        private readonly List<MeshTriUV> _hMeshTriUVs = new List<MeshTriUV>();
        private readonly List<int> _hTriMaterialIndex = new List<int>();
        private readonly List<MaterialRecord> _hMaterials = new List<MaterialRecord>();
        private readonly List<TexInfo> _hTexInfos = new List<TexInfo>();
        private readonly List<RGBA32> _hTexels = new List<RGBA32>();

        private MemoryBuffer1D<TLASNode, Stride1D.Dense> _dTLASNodes;
        private MemoryBuffer1D<int, Stride1D.Dense> _dTLASInstanceIndices;
        private MemoryBuffer1D<InstanceRecord, Stride1D.Dense> _dInstances;

        private MemoryBuffer1D<BLASNode, Stride1D.Dense> _dBLASNodes;

        private MemoryBuffer1D<int, Stride1D.Dense> _dSpherePrimIndices;
        private MemoryBuffer1D<Sphere, Stride1D.Dense> _dSpheres;

        private MemoryBuffer1D<int, Stride1D.Dense> _dTriPrimIndices;
        private MemoryBuffer1D<Float3, Stride1D.Dense> _dMeshPositions;
        private MemoryBuffer1D<MeshTri, Stride1D.Dense> _dMeshTris;

        private MemoryBuffer1D<Float2, Stride1D.Dense> _dMeshTexcoords;
        private MemoryBuffer1D<MeshTriUV, Stride1D.Dense> _dMeshTriUVs;
        private MemoryBuffer1D<int, Stride1D.Dense> _dTriMaterialIndex;
        private MemoryBuffer1D<MaterialRecord, Stride1D.Dense> _dMaterials;
        private MemoryBuffer1D<RGBA32, Stride1D.Dense> _dTexels;
        private MemoryBuffer1D<TexInfo, Stride1D.Dense> _dTexInfos;

        public Scene(CudaAccelerator cuda)
        {
            _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda));
            UploadAll();
        }

        public ArrayView<TLASNode> TLASNodesView => _dTLASNodes.View;
        public ArrayView<int> TLASInstanceIndicesView => _dTLASInstanceIndices.View;
        public ArrayView<InstanceRecord> InstancesView => _dInstances.View;
        public ArrayView<BLASNode> BLASNodesView => _dBLASNodes.View;
        public ArrayView<int> SpherePrimIndicesView => _dSpherePrimIndices.View;
        public ArrayView<Sphere> SpheresView => _dSpheres.View;
        public ArrayView<int> TriPrimIndicesView => _dTriPrimIndices.View;
        public ArrayView<Float3> MeshPositionsView => _dMeshPositions.View;
        public ArrayView<MeshTri> MeshTrisView => _dMeshTris.View;

        public ArrayView<Float2> MeshTexcoordsView => _dMeshTexcoords.View;
        public ArrayView<MeshTriUV> MeshTriUVsView => _dMeshTriUVs.View;
        public ArrayView<int> TriMaterialIndexView => _dTriMaterialIndex.View;
        public ArrayView<MaterialRecord> MaterialsView => _dMaterials.View;
        public ArrayView<RGBA32> TexelsView => _dTexels.View;
        public ArrayView<TexInfo> TexInfosView => _dTexInfos.View;

        public void BuildDefaultScene()
        {
            _hBLASNodes.Clear();
            _hSpherePrimIndices.Clear();
            _hSpheres.Clear();
            _hTriPrimIndices.Clear();
            _hMeshPositions.Clear();
            _hMeshTris.Clear();
            _hMeshTexcoords.Clear();
            _hMeshTriUVs.Clear();
            _hTriMaterialIndex.Clear();
            _hMaterials.Clear();
            _hTexInfos.Clear();
            _hTexels.Clear();

            int AddCheckerTexture(int w, int h, int step, RGBA32 c0, RGBA32 c1)
            {
                int offset = _hTexels.Count;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        bool a = (((x / step) + (y / step)) & 1) == 0;
                        _hTexels.Add(a ? c0 : c1);
                    }
                _hTexInfos.Add(new TexInfo { Offset = offset, Width = w, Height = h });
                return _hTexInfos.Count - 1;
            }

            int checker0 = AddCheckerTexture(256, 256, 16, new RGBA32 { R = 255, G = 255, B = 255, A = 255 }, new RGBA32 { R = 20, G = 20, B = 20, A = 255 });
            int checker1 = AddCheckerTexture(256, 256, 8, new RGBA32 { R = 40, G = 40, B = 200, A = 255 }, new RGBA32 { R = 200, G = 200, B = 40, A = 255 });

            var matGround = new MaterialRecord { Kd = new Float3(1f, 1f, 1f), HasDiffuseMap = 1, DiffuseTexIndex = checker0, Shading = MaterialRecord.SHADING_LAMBERT, IOR = 1f, HasAlphaMap = 0, AlphaTexIndex = -1, AlphaCutoff = 0.5f, TwoSided = 0 };
            var matRed = new MaterialRecord { Kd = new Float3(0.8f, 0.3f, 0.3f), HasDiffuseMap = 0, DiffuseTexIndex = -1, Shading = MaterialRecord.SHADING_LAMBERT, IOR = 1f, HasAlphaMap = 0, AlphaTexIndex = -1, AlphaCutoff = 0.5f, TwoSided = 0 };
            var matGreen = new MaterialRecord { Kd = new Float3(0.3f, 0.8f, 0.3f), HasDiffuseMap = 0, DiffuseTexIndex = -1, Shading = MaterialRecord.SHADING_LAMBERT, IOR = 1f, HasAlphaMap = 0, AlphaTexIndex = -1, AlphaCutoff = 0.5f, TwoSided = 0 };
            var matTex = new MaterialRecord { Kd = new Float3(1f, 1f, 1f), HasDiffuseMap = 1, DiffuseTexIndex = checker1, Shading = MaterialRecord.SHADING_LAMBERT, IOR = 1f, HasAlphaMap = 0, AlphaTexIndex = -1, AlphaCutoff = 0.5f, TwoSided = 0 };
            var matWhite = new MaterialRecord { Kd = new Float3(1f, 1f, 1f), HasDiffuseMap = 0, DiffuseTexIndex = -1, Shading = MaterialRecord.SHADING_LAMBERT, IOR = 1f, HasAlphaMap = 0, AlphaTexIndex = -1, AlphaCutoff = 0.5f, TwoSided = 0 };

            int ground = AddSphere(new Sphere { center = new Float3(0f, -1000.5f, 0f), radius = 1000f, albedo = new Float3(1f, 1f, 1f), material = matGround, shading = Sphere.SHADING_LAMBERT, ior = 1f });
            int s0 = AddSphere(new Sphere { center = new Float3(-0.9f, 0.5f, -0.2f), radius = 0.5f, albedo = new Float3(0.8f, 0.3f, 0.3f), material = matRed, shading = Sphere.SHADING_LAMBERT, ior = 1f });
            int s1 = AddSphere(new Sphere { center = new Float3(0.9f, 0.35f, 0.2f), radius = 0.35f, albedo = new Float3(0.3f, 0.8f, 0.3f), material = matGreen, shading = Sphere.SHADING_LAMBERT, ior = 1f });
            int s2 = AddSphere(new Sphere { center = new Float3(0.0f, 0.75f, 0.6f), radius = 0.75f, albedo = new Float3(1f, 1f, 1f), material = matTex, shading = Sphere.SHADING_LAMBERT, ior = 1f });
            int sMirror = AddSphere(new Sphere { center = new Float3(-1.8f, 0.5f, 0.8f), radius = 0.5f, albedo = new Float3(1f, 1f, 1f), material = matWhite, shading = Sphere.SHADING_MIRROR, ior = 1f });
            int sGlass = AddSphere(new Sphere { center = new Float3(1.8f, 0.5f, -0.8f), radius = 0.5f, albedo = new Float3(1f, 1f, 1f), material = matWhite, shading = Sphere.SHADING_GLASS, ior = 1.5f });

            var inst = new List<InstanceRecord>
            {
                BuildSphereInstance(new int[] { ground },  Affine3x4.Identity()),
                BuildSphereInstance(new int[] { s0 },      Affine3x4.Identity()),
                BuildSphereInstance(new int[] { s1 },      Affine3x4.Identity()),
                BuildSphereInstance(new int[] { s2 },      Affine3x4.Identity()),
                BuildSphereInstance(new int[] { sMirror }, Affine3x4.Identity()),
                BuildSphereInstance(new int[] { sGlass },  Affine3x4.Identity()),
            };

            _hInstances = inst.ToArray();

            TryAddSponzaFromKnownLocations();

            RebuildTLAS(_hInstances, out _hTLASNodes, out _hTLASInstanceIndices);
        }

        public void LoadObjInstance(string objPath, Affine3x4 objectToWorld, float uniformScale = 1f)
        {
            if (string.IsNullOrWhiteSpace(objPath) || !File.Exists(objPath))
                throw new FileNotFoundException("OBJ file not found.", objPath);

            MeshHost mesh = MeshLoaderOBJ.Load(objPath, uniformScale, flipWinding: false);

            int baseVertex = _hMeshPositions.Count;
            int baseTri = _hMeshTris.Count;
            int baseUV = _hMeshTexcoords.Count;
            int baseMat = _hMaterials.Count;

            _hMeshPositions.AddRange(mesh.Positions);
            _hMeshTexcoords.AddRange(mesh.Texcoords);

            for (int i = 0; i < mesh.Triangles.Count; i++)
            {
                MeshTri t = mesh.Triangles[i];
                t.i0 += baseVertex;
                t.i1 += baseVertex;
                t.i2 += baseVertex;
                _hMeshTris.Add(t);

                var tuv = mesh.TriUVs[i];
                tuv.t0 += baseUV;
                tuv.t1 += baseUV;
                tuv.t2 += baseUV;
                _hMeshTriUVs.Add(tuv);

                int matIndexLocal = (i < mesh.TriMaterialIndex.Count) ? mesh.TriMaterialIndex[i] : 0;
                int matIndexGlobal = baseMat + matIndexLocal;
                _hTriMaterialIndex.Add(matIndexGlobal);

                _hTriPrimIndices.Add(baseTri + i);
            }

            // Append materials, remapping each material's texture index to a flattened texel array
            // We store each texture serially and emit TexInfo with offset/size for sampling
            int texBase = _hTexels.Count;
            var localMatCount = mesh.Materials.Count;
            var matRemapped = new List<MaterialRecord>(localMatCount);
            for (int i = 0; i < localMatCount; i++)
            {
                var m = mesh.Materials[i];

                // Remap diffuse (map_Kd)
                if (m.HasDiffuseMap != 0 && m.DiffuseTexIndex >= 0 && m.DiffuseTexIndex < mesh.Textures.Count)
                {
                    var src = mesh.Textures[m.DiffuseTexIndex];
                    int start = _hTexels.Count;
                    for (int p = 0; p < src.BGRA.Length; p += 4)
                        _hTexels.Add(new RGBA32 { B = src.BGRA[p + 0], G = src.BGRA[p + 1], R = src.BGRA[p + 2], A = src.BGRA[p + 3] });
                    int texIndexGlobal = _hTexInfos.Count;
                    _hTexInfos.Add(new TexInfo { Offset = start, Width = src.Width, Height = src.Height });
                    m.DiffuseTexIndex = texIndexGlobal;
                    m.HasDiffuseMap = 1;
                }
                else
                {
                    m.HasDiffuseMap = 0;
                    m.DiffuseTexIndex = -1;
                }

                // NEW: remap alpha (map_d)
                if (m.HasAlphaMap != 0 && m.AlphaTexIndex >= 0 && m.AlphaTexIndex < mesh.Textures.Count)
                {
                    var srcA = mesh.Textures[m.AlphaTexIndex];
                    int startA = _hTexels.Count;
                    for (int p = 0; p < srcA.BGRA.Length; p += 4)
                        _hTexels.Add(new RGBA32 { B = srcA.BGRA[p + 0], G = srcA.BGRA[p + 1], R = srcA.BGRA[p + 2], A = srcA.BGRA[p + 3] });
                    int texIndexGlobalA = _hTexInfos.Count;
                    _hTexInfos.Add(new TexInfo { Offset = startA, Width = srcA.Width, Height = srcA.Height });
                    m.AlphaTexIndex = texIndexGlobalA;
                    m.HasAlphaMap = 1;
                }
                else
                {
                    m.HasAlphaMap = 0;
                    m.AlphaTexIndex = -1;
                }

                matRemapped.Add(m);
            }
            _hMaterials.AddRange(matRemapped);

            MeshGlobal.SetTris(_hMeshTris);

            int blasStart = _hBLASNodes.Count;
            BuildBLAS_Triangles(_hBLASNodes, _hTriPrimIndices, baseTri, mesh.Triangles.Count, _hMeshPositions);
            int blasCount = _hBLASNodes.Count - blasStart;

            ComputeMeshBounds(mesh.Positions, mesh.Triangles, out var bmin, out var bmax);
            TransformAABB(objectToWorld, bmin, bmax, out var wmin, out var wmax);

            Affine3x4 worldToObject = InvertRigidOrUniform(objectToWorld, out float uniScale);

            InstanceRecord instRec = default;
            instRec.type = BlasType.TriMesh;
            instRec.blasRoot = blasStart;
            instRec.blasNodeCount = blasCount;
            instRec.primIndexFirst = baseTri;
            instRec.primIndexCount = mesh.Triangles.Count;
            instRec.objectToWorld = objectToWorld;
            instRec.worldToObject = worldToObject;
            instRec.uniformScale = uniScale;
            instRec.worldBoundsMin = wmin;
            instRec.worldBoundsMax = wmax;

            var instList = new List<InstanceRecord>(_hInstances) { instRec };
            _hInstances = instList.ToArray();

            RebuildTLAS(_hInstances, out _hTLASNodes, out _hTLASInstanceIndices);
        }

        public void UploadAll()
        {
            _dTLASNodes.DisposeIfValid(); _dTLASNodes = AllocateOrEmpty(_hTLASNodes ?? Array.Empty<TLASNode>());
            _dTLASInstanceIndices.DisposeIfValid(); _dTLASInstanceIndices = AllocateOrEmpty(_hTLASInstanceIndices ?? Array.Empty<int>());
            _dInstances.DisposeIfValid(); _dInstances = AllocateOrEmpty(_hInstances ?? Array.Empty<InstanceRecord>());

            _dBLASNodes.DisposeIfValid(); _dBLASNodes = AllocateOrEmpty((_hBLASNodes != null ? _hBLASNodes.ToArray() : Array.Empty<BLASNode>()));

            _dSpherePrimIndices.DisposeIfValid(); _dSpherePrimIndices = AllocateOrEmpty((_hSpherePrimIndices != null ? _hSpherePrimIndices.ToArray() : Array.Empty<int>()));
            _dSpheres.DisposeIfValid(); _dSpheres = AllocateOrEmpty((_hSpheres != null ? _hSpheres.ToArray() : Array.Empty<Sphere>()));

            _dTriPrimIndices.DisposeIfValid(); _dTriPrimIndices = AllocateOrEmpty((_hTriPrimIndices != null ? _hTriPrimIndices.ToArray() : Array.Empty<int>()));
            _dMeshPositions.DisposeIfValid(); _dMeshPositions = AllocateOrEmpty((_hMeshPositions != null ? _hMeshPositions.ToArray() : Array.Empty<Float3>()));
            _dMeshTris.DisposeIfValid(); _dMeshTris = AllocateOrEmpty((_hMeshTris != null ? _hMeshTris.ToArray() : Array.Empty<MeshTri>()));

            _dMeshTexcoords.DisposeIfValid(); _dMeshTexcoords = AllocateOrEmpty((_hMeshTexcoords != null ? _hMeshTexcoords.ToArray() : Array.Empty<Float2>()));
            _dMeshTriUVs.DisposeIfValid(); _dMeshTriUVs = AllocateOrEmpty((_hMeshTriUVs != null ? _hMeshTriUVs.ToArray() : Array.Empty<MeshTriUV>()));
            _dTriMaterialIndex.DisposeIfValid(); _dTriMaterialIndex = AllocateOrEmpty((_hTriMaterialIndex != null ? _hTriMaterialIndex.ToArray() : Array.Empty<int>()));
            _dMaterials.DisposeIfValid(); _dMaterials = AllocateOrEmpty((_hMaterials != null ? _hMaterials.ToArray() : Array.Empty<MaterialRecord>()));
            _dTexels.DisposeIfValid(); _dTexels = AllocateOrEmpty((_hTexels != null ? _hTexels.ToArray() : Array.Empty<RGBA32>()));
            _dTexInfos.DisposeIfValid(); _dTexInfos = AllocateOrEmpty((_hTexInfos != null ? _hTexInfos.ToArray() : Array.Empty<TexInfo>()));
        }

        public void GetDeviceViews(
            out ArrayView<TLASNode> tlasNodes,
            out ArrayView<int> tlasInstanceIndices,
            out ArrayView<InstanceRecord> instances,
            out ArrayView<BLASNode> blasNodes,
            out ArrayView<int> spherePrimIdx,
            out ArrayView<Sphere> spheres,
            out ArrayView<int> triPrimIdx,
            out ArrayView<Float3> meshPositions,
            out ArrayView<MeshTri> meshTris,
            out ArrayView<Float2> meshTexcoords,
            out ArrayView<MeshTriUV> meshTriUVs,
            out ArrayView<int> triMatIndex,
            out ArrayView<MaterialRecord> materials,
            out ArrayView<RGBA32> texels,
            out ArrayView<TexInfo> texInfos)
        {
            tlasNodes = _dTLASNodes.View;
            tlasInstanceIndices = _dTLASInstanceIndices.View;
            instances = _dInstances.View;
            blasNodes = _dBLASNodes.View;
            spherePrimIdx = _dSpherePrimIndices.View;
            spheres = _dSpheres.View;
            triPrimIdx = _dTriPrimIndices.View;
            meshPositions = _dMeshPositions.View;
            meshTris = _dMeshTris.View;
            meshTexcoords = _dMeshTexcoords.View;
            meshTriUVs = _dMeshTriUVs.View;
            triMatIndex = _dTriMaterialIndex.View;
            materials = _dMaterials.View;
            texels = _dTexels.View;
            texInfos = _dTexInfos.View;
        }

        private int AddSphere(Sphere s)
        {
            int id = _hSpheres.Count;
            _hSpheres.Add(s);
            _hSpherePrimIndices.Add(id);
            return id;
        }

        private InstanceRecord BuildSphereInstance(int[] sphereIds, Affine3x4 objectToWorld)
        {
            Float3 bmin = new Float3(float.MaxValue, float.MaxValue, float.MaxValue);
            Float3 bmax = new Float3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < sphereIds.Length; i++)
            {
                Sphere s = _hSpheres[sphereIds[i]];
                bmin = Float3.Min(bmin, new Float3(s.center.X - s.radius, s.center.Y - s.radius, s.center.Z - s.radius));
                bmax = Float3.Max(bmax, new Float3(s.center.X + s.radius, s.center.Y + s.radius, s.center.Z + s.radius));
            }

            int primStart = sphereIds[0];
            int primCount = sphereIds.Length;

            int blasStart = _hBLASNodes.Count;
            BuildBLAS_Spheres(_hBLASNodes, _hSpherePrimIndices, primStart, primCount, _hSpheres);
            int blasCount = _hBLASNodes.Count - blasStart;

            TransformAABB(objectToWorld, bmin, bmax, out var wmin, out var wmax);
            Affine3x4 worldToObject = InvertRigidOrUniform(objectToWorld, out float uniScale);

            InstanceRecord inst = default;
            inst.type = BlasType.SphereSet;
            inst.blasRoot = blasStart;
            inst.blasNodeCount = blasCount;
            inst.primIndexFirst = primStart;
            inst.primIndexCount = primCount;
            inst.objectToWorld = objectToWorld;
            inst.worldToObject = worldToObject;
            inst.uniformScale = uniScale;
            inst.worldBoundsMin = wmin;
            inst.worldBoundsMax = wmax;
            return inst;
        }

        private void RebuildTLAS(InstanceRecord[] instances, out TLASNode[] nodes, out int[] instanceIndices)
        {
            int n = instances.Length;
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;

            var outNodes = new List<TLASNode>(2 * n);
            BuildTLASNodeRecursive(instances, idx, 0, n, outNodes, -1);
            nodes = outNodes.ToArray();
            instanceIndices = idx;
        }

        private MemoryBuffer1D<T, Stride1D.Dense> AllocateOrEmpty<T>(T[] src) where T : unmanaged
        {
            if (src != null && src.Length > 0)
                return _cuda.Allocate1D(src);
            var buf = _cuda.Allocate1D<T>(1);
            buf.MemSetToZero();
            return buf;
        }

        // ----- BLAS/TLAS builders (unchanged from your version) -----

        private static void BuildBLAS_Spheres(List<BLASNode> outBLAS, List<int> primIdx, int primStart, int primCount, List<Sphere> spheres)
        {
            int[] idx = new int[primCount];
            for (int i = 0; i < primCount; i++) idx[i] = primStart + i;

            Float3[] bmin = new Float3[primCount];
            Float3[] bmax = new Float3[primCount];
            for (int i = 0; i < primCount; i++)
            {
                Sphere s = spheres[primIdx[primStart + i]];
                bmin[i] = new Float3(s.center.X - s.radius, s.center.Y - s.radius, s.center.Z - s.radius);
                bmax[i] = new Float3(s.center.X + s.radius, s.center.Y + s.radius, s.center.Z + s.radius);
            }

            BuildBLASNodeRecursive(outBLAS, primIdx, idx, 0, primCount, bmin, bmax, -1, spheres, null);
        }

        private static void BuildBLAS_Triangles(List<BLASNode> outBLAS, List<int> primIdx, int primStart, int primCount, List<Float3> positions)
        {
            int[] idx = new int[primCount];
            for (int i = 0; i < primCount; i++) idx[i] = primStart + i;
            BuildBLASNodeRecursive(outBLAS, primIdx, idx, 0, primCount, null, null, -1, null, positions);
        }

        private static int BuildBLASNodeRecursive(List<BLASNode> outBLAS, List<int> primIdx, int[] idx, int start, int count, Float3[] bminPre, Float3[] bmaxPre, int parentSkip, List<Sphere> spheresOrNull, List<Float3> positionsOrNull)
        {
            int nodeIndex = outBLAS.Count;
            BLASNode node = default;
            node.first = -1; node.count = 0; node.left = -1; node.right = -1; node.skipIndex = parentSkip;

            Float3 nbMin = new Float3(float.MaxValue, float.MaxValue, float.MaxValue);
            Float3 nbMax = new Float3(float.MinValue, float.MinValue, float.MinValue);
            if (bminPre != null)
            {
                for (int i = start; i < start + count; i++)
                {
                    nbMin = Float3.Min(nbMin, bminPre[i]);
                    nbMax = Float3.Max(nbMax, bmaxPre[i]);
                }
            }
            else
            {
                for (int i = start; i < start + count; i++)
                {
                    int triIndex = primIdx[idx[i]];
                    BoundsOfTriangle(triIndex, positionsOrNull, out var mn, out var mx);
                    nbMin = Float3.Min(nbMin, mn);
                    nbMax = Float3.Max(nbMax, mx);
                }
            }

            node.boundsMin = nbMin;
            node.boundsMax = nbMax;
            outBLAS.Add(node);

            const int LeafThreshold = 4;
            if (count <= LeafThreshold)
            {
                int leafStart = primIdx.Count;
                for (int i = start; i < start + count; i++) primIdx.Add(primIdx[idx[i]]);
                BLASNode leaf = outBLAS[nodeIndex];
                leaf.first = leafStart; leaf.count = count; leaf.skipIndex = parentSkip;
                outBLAS[nodeIndex] = leaf;
                return nodeIndex;
            }

            Float3 extent = nbMax - nbMin;
            int axis = 0;
            if (extent.Y > extent.X && extent.Y >= extent.Z) axis = 1;
            else if (extent.Z > extent.X && extent.Z >= extent.Y) axis = 2;

            if (spheresOrNull != null)
                Array.Sort(idx, start, count, new BLASPrimComparatorSpheres(axis, primIdx, spheresOrNull));
            else
                Array.Sort(idx, start, count, new BLASPrimComparatorTris(axis, primIdx, positionsOrNull));

            int mid = start + (count >> 1);

            int rightRoot = BuildBLASNodeRecursive(outBLAS, primIdx, idx, mid, count - (mid - start), bminPre, bmaxPre, parentSkip, spheresOrNull, positionsOrNull);
            int leftRoot = BuildBLASNodeRecursive(outBLAS, primIdx, idx, start, mid - start, bminPre, bmaxPre, rightRoot, spheresOrNull, positionsOrNull);

            BLASNode inner = outBLAS[nodeIndex];
            inner.left = leftRoot; inner.right = rightRoot; inner.skipIndex = parentSkip;
            outBLAS[nodeIndex] = inner;

            return nodeIndex;
        }

        private static int BuildTLASNodeRecursive(InstanceRecord[] inst, int[] idx, int start, int count, List<TLASNode> outNodes, int parentSkip)
        {
            int nodeIndex = outNodes.Count;
            TLASNode node = default;
            node.first = -1; node.count = 0; node.left = -1; node.right = -1; node.skipIndex = parentSkip;

            Float3 nbMin = new Float3(float.MaxValue, float.MaxValue, float.MaxValue);
            Float3 nbMax = new Float3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = start; i < start + count; i++)
            {
                InstanceRecord r = inst[idx[i]];
                nbMin = Float3.Min(nbMin, r.worldBoundsMin);
                nbMax = Float3.Max(nbMax, r.worldBoundsMax);
            }
            node.boundsMin = nbMin; node.boundsMax = nbMax;
            outNodes.Add(node);

            const int LeafThreshold = 2;
            if (count <= LeafThreshold)
            {
                TLASNode leaf = outNodes[nodeIndex];
                leaf.first = start; leaf.count = count; leaf.skipIndex = parentSkip;
                outNodes[nodeIndex] = leaf;
                return nodeIndex;
            }

            Float3 extent = nbMax - nbMin;
            int axis = 0;
            if (extent.Y > extent.X && extent.Y >= extent.Z) axis = 1;
            else if (extent.Z > extent.X && extent.Z >= extent.Y) axis = 2;

            Array.Sort(idx, start, count, new TLASInstComparator(axis, inst));

            int mid = start + (count >> 1);
            int rightRoot = BuildTLASNodeRecursive(inst, idx, mid, count - (mid - start), outNodes, parentSkip);
            int leftRoot = BuildTLASNodeRecursive(inst, idx, start, mid - start, outNodes, rightRoot);

            TLASNode inner = outNodes[nodeIndex];
            inner.left = leftRoot; inner.right = rightRoot; inner.skipIndex = parentSkip;
            outNodes[nodeIndex] = inner;
            return nodeIndex;
        }

        private readonly struct BLASPrimComparatorSpheres : IComparer<int>
        {
            private readonly int _axis;
            private readonly List<int> _primIdx;
            private readonly List<Sphere> _spheres;
            public BLASPrimComparatorSpheres(int axis, List<int> primIdx, List<Sphere> spheres) { _axis = axis; _primIdx = primIdx; _spheres = spheres; }
            public int Compare(int a, int b)
            {
                int ia = _primIdx[a], ib = _primIdx[b];
                Sphere sa = _spheres[ia], sb = _spheres[ib];
                float ca = _axis == 0 ? sa.center.X : (_axis == 1 ? sa.center.Y : sa.center.Z);
                float cb = _axis == 0 ? sb.center.X : (_axis == 1 ? sb.center.Y : sb.center.Z);
                if (ca < cb) return -1; if (ca > cb) return 1; return 0;
            }
        }

        private readonly struct BLASPrimComparatorTris : IComparer<int>
        {
            private readonly int _axis;
            private readonly List<int> _primIdx;
            private readonly List<Float3> _positions;
            public BLASPrimComparatorTris(int axis, List<int> primIdx, List<Float3> positions) { _axis = axis; _primIdx = primIdx; _positions = positions; }
            public int Compare(int a, int b)
            {
                int ia = _primIdx[a], ib = _primIdx[b];
                Float3 ca = CenterOfTriangle(ia, _positions);
                Float3 cb = CenterOfTriangle(ib, _positions);
                float va = _axis == 0 ? ca.X : (_axis == 1 ? ca.Y : ca.Z);
                float vb = _axis == 0 ? cb.X : (_axis == 1 ? cb.Y : cb.Z);
                if (va < vb) return -1; if (va > vb) return 1; return 0;
            }
        }

        private readonly struct TLASInstComparator : IComparer<int>
        {
            private readonly int _axis;
            private readonly InstanceRecord[] _inst;
            public TLASInstComparator(int axis, InstanceRecord[] inst) { _axis = axis; _inst = inst; }
            public int Compare(int a, int b)
            {
                Float3 ca = Float3.Center(_inst[a].worldBoundsMin, _inst[a].worldBoundsMax);
                Float3 cb = Float3.Center(_inst[b].worldBoundsMin, _inst[b].worldBoundsMax);
                float va = _axis == 0 ? ca.X : (_axis == 1 ? ca.Y : ca.Z);
                float vb = _axis == 0 ? cb.X : (_axis == 1 ? cb.Y : cb.Z);
                if (va < vb) return -1; if (va > vb) return 1; return 0;
            }
        }

        private static void TransformAABB(Affine3x4 m, Float3 bmin, Float3 bmax, out Float3 outMin, out Float3 outMax)
        {
            Float3[] c = new Float3[8];
            c[0] = new Float3(bmin.X, bmin.Y, bmin.Z);
            c[1] = new Float3(bmax.X, bmin.Y, bmin.Z);
            c[2] = new Float3(bmin.X, bmax.Y, bmin.Z);
            c[3] = new Float3(bmin.X, bmin.Y, bmax.Z);
            c[4] = new Float3(bmax.X, bmax.Y, bmin.Z);
            c[5] = new Float3(bmin.X, bmax.Y, bmax.Z);
            c[6] = new Float3(bmax.X, bmin.Y, bmax.Z);
            c[7] = new Float3(bmax.X, bmax.Y, bmax.Z);
            Float3 mn = new Float3(float.MaxValue, float.MaxValue, float.MaxValue);
            Float3 mx = new Float3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < 8; i++)
            {
                Float3 w = TransformPoint(m, c[i]);
                mn = Float3.Min(mn, w);
                mx = Float3.Max(mx, w);
            }
            outMin = mn; outMax = mx;
        }

        private static void ComputeMeshBounds(List<Float3> pos, List<MeshTri> tris, out Float3 bmin, out Float3 bmax)
        {
            bmin = new Float3(float.MaxValue, float.MaxValue, float.MaxValue);
            bmax = new Float3(float.MinValue, float.MinValue, float.MinValue);
            for (int i = 0; i < tris.Count; i++)
            {
                MeshTri t = tris[i];
                Float3 v0 = pos[t.i0];
                Float3 v1 = pos[t.i1];
                Float3 v2 = pos[t.i2];
                bmin = Float3.Min(bmin, Float3.Min(v0, Float3.Min(v1, v2)));
                bmax = Float3.Max(bmax, Float3.Max(v0, Float3.Max(v1, v2)));
            }
        }

        private static void BoundsOfTriangle(int triIndex, List<Float3> positions, out Float3 mn, out Float3 mx)
        {
            MeshTri tri = MeshGlobal.ResolveTri(triIndex);
            Float3 v0 = positions[tri.i0];
            Float3 v1 = positions[tri.i1];
            Float3 v2 = positions[tri.i2];
            mn = Float3.Min(v0, Float3.Min(v1, v2));
            mx = Float3.Max(v0, Float3.Max(v1, v2));
        }

        private static Float3 CenterOfTriangle(int triIndex, List<Float3> positions)
        {
            MeshTri tri = MeshGlobal.ResolveTri(triIndex);
            Float3 v0 = positions[tri.i0];
            Float3 v1 = positions[tri.i1];
            Float3 v2 = positions[tri.i2];
            return new Float3((v0.X + v1.X + v2.X) / 3f, (v0.Y + v1.Y + v2.Y) / 3f, (v0.Z + v1.Z + v2.Z) / 3f);
        }

        private static Affine3x4 InvertRigidOrUniform(Affine3x4 m, out float uniformScale)
        {
            float sx = Float3.Length(new Float3(m.m00, m.m10, m.m20));
            float sy = Float3.Length(new Float3(m.m01, m.m11, m.m21));
            float sz = Float3.Length(new Float3(m.m02, m.m12, m.m22));
            uniformScale = (sx + sy + sz) / 3f;
            float inv = uniformScale > 0f ? 1f / uniformScale : 1f;

            Float3 r0 = Float3.Normalize(new Float3(m.m00, m.m10, m.m20));
            Float3 r1 = Float3.Normalize(new Float3(m.m01, m.m11, m.m21));
            Float3 r2 = Float3.Normalize(new Float3(m.m02, m.m12, m.m22));

            Affine3x4 invM = default;
            invM.m00 = r0.X * inv; invM.m01 = r1.X * inv; invM.m02 = r2.X * inv; invM.m03 = 0f;
            invM.m10 = r0.Y * inv; invM.m11 = r1.Y * inv; invM.m12 = r2.Y * inv; invM.m13 = 0f;
            invM.m20 = r0.Z * inv; invM.m21 = r1.Z * inv; invM.m22 = r2.Z * inv; invM.m23 = 0f;

            Float3 t = new Float3(m.m03, m.m13, m.m23);
            Float3 it = TransformVector(invM, t) * -1f;
            invM.m03 = it.X; invM.m13 = it.Y; invM.m23 = it.Z;

            return invM;
        }

        private static Float3 TransformPoint(Affine3x4 m, Float3 p)
        {
            return new Float3(m.m00 * p.X + m.m01 * p.Y + m.m02 * p.Z + m.m03,
                              m.m10 * p.X + m.m11 * p.Y + m.m12 * p.Z + m.m13,
                              m.m20 * p.X + m.m21 * p.Y + m.m22 * p.Z + m.m23);
        }

        private static Float3 TransformVector(Affine3x4 m, Float3 v)
        {
            return new Float3(m.m00 * v.X + m.m01 * v.Y + m.m02 * v.Z,
                              m.m10 * v.X + m.m11 * v.Y + m.m12 * v.Z,
                              m.m20 * v.X + m.m21 * v.Y + m.m22 * v.Z);
        }

        private bool TryAddSponzaFromKnownLocations()
        {
            string baseDir = AppContext.BaseDirectory ?? "";
            string curDir = Directory.GetCurrentDirectory();
            string[] candidates = new[]
            {
                Path.Combine(baseDir, "Sponza", "sponza.obj"),
                Path.Combine(curDir, "Sponza", "sponza.obj"),
                @"C:\Users\alec\source\repos\ILGPU_Raytracing\ILGPU_Raytracing\bin\Debug\net8.0\Assets\Sponza\sponza.obj"
            };

            foreach (var p in candidates)
            {
                if (File.Exists(p))
                {
                    LoadObjInstance(p, Affine3x4.Identity(), 0.01f);
                    return true;
                }
            }
            return false;
        }

        public void Dispose()
        {
            _dTLASNodes.DisposeIfValid();
            _dTLASInstanceIndices.DisposeIfValid();
            _dInstances.DisposeIfValid();
            _dBLASNodes.DisposeIfValid();
            _dSpherePrimIndices.DisposeIfValid();
            _dSpheres.DisposeIfValid();
            _dTriPrimIndices.DisposeIfValid();
            _dMeshPositions.DisposeIfValid();
            _dMeshTris.DisposeIfValid();
            _dMeshTexcoords.DisposeIfValid();
            _dMeshTriUVs.DisposeIfValid();
            _dTriMaterialIndex.DisposeIfValid();
            _dMaterials.DisposeIfValid();
            _dTexels.DisposeIfValid();
            _dTexInfos.DisposeIfValid();
        }
    }

    public static class MeshGlobal
    {
        private static List<MeshTri> _trisRef = new List<MeshTri>();
        public static void SetTris(List<MeshTri> tris) { _trisRef = tris; }
        public static MeshTri ResolveTri(int globalTriIndex) { return _trisRef[globalTriIndex]; }
    }

    public enum BlasType { SphereSet = 1, TriMesh = 2 }

    public struct TLASNode
    {
        public Float3 boundsMin;
        public Float3 boundsMax;
        public int left;
        public int right;
        public int first;
        public int count;
        public int skipIndex;
    }

    public struct InstanceRecord
    {
        public BlasType type;
        public int blasRoot;
        public int blasNodeCount;
        public int primIndexFirst;
        public int primIndexCount;
        public Affine3x4 objectToWorld;
        public Affine3x4 worldToObject;
        public float uniformScale;
        public Float3 worldBoundsMin;
        public Float3 worldBoundsMax;
    }

    public struct BLASNode
    {
        public Float3 boundsMin;
        public Float3 boundsMax;
        public int left;
        public int right;
        public int first;
        public int count;
        public int skipIndex;
    }

    public struct MeshTri { public int i0, i1, i2; }

    public struct RGBA32 { public byte R, G, B, A; }

    public struct TexInfo { public int Offset, Width, Height; }

    public static class BufferUtils
    {
        public static void DisposeIfValid(this MemoryBuffer buf) { if (buf != null) buf.Dispose(); }
        public static void DisposeIfValid<T, TStride>(this MemoryBuffer1D<T, TStride> buf)
            where T : unmanaged where TStride : struct, IStride1D
        { if (buf != null && buf.IsValid) buf.Dispose(); }
    }
}
