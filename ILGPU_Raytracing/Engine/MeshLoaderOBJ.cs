// ==============================
// File: Engine/MeshLoaderOBJ.cs
// - Strict RGBA handling (no premultiplication)
// - NO fallbacks: missing textures are skipped (flags cleared)
// - map_d (alpha mask) controls cutouts; TwoSided default for alpha mats
// - Detailed Console.WriteLine diagnostics
// ==============================

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using ILGPU.Algorithms;
using ILGPU;
using System.Runtime.CompilerServices;

namespace ILGPU_Raytracing.Engine
{
    public sealed class MeshHost
    {
        public readonly List<Float3> Positions = new List<Float3>();
        public readonly List<MeshTri> Triangles = new List<MeshTri>();

        public readonly List<Float2> Texcoords = new List<Float2>();
        public readonly List<MeshTriUV> TriUVs = new List<MeshTriUV>();
        public readonly List<int> TriMaterialIndex = new List<int>();
        public readonly List<MaterialRecord> Materials = new List<MaterialRecord>();
        public readonly List<TextureSrc> Textures = new List<TextureSrc>();
    }

    public struct Float2 { public float X, Y; public Float2(float x, float y) { X = x; Y = y; } }
    public struct MeshTriUV { public int t0, t1, t2; }

    public struct TextureSrc
    {
        public string Path;
        public int Width, Height;
        public byte[] BGRA; // straight alpha, not premultiplied
    }

    // Keep size multiple of 16B. This layout is 48B (OK).
    public struct MaterialRecord
    {
        public const int SHADING_LAMBERT = 0;
        public const int SHADING_MIRROR = 1;
        public const int SHADING_GLASS = 2;

        public Float3 Kd;           // 12
        public int HasDiffuseMap;   // 16
        public int DiffuseTexIndex; // 20

        public int Shading;         // 24
        public float IOR;           // 28

        // Alpha cutout pipeline
        public int HasAlphaMap;     // 32 (from map_d)
        public int AlphaTexIndex;   // 36
        public int TwoSided;        // 40 (flip normals on backface)
        public float AlphaCutoff;   // 44 (default 0.5)
        // total: 48 bytes
    }

    public static class MeshLoaderOBJ
    {
        public static MeshHost Load(string path, float scale = 1f, bool flipWinding = true)
        {
            Console.WriteLine("[OBJ] Loading '{0}' (scale={1}, flipWinding={2})", path, scale, flipWinding);
            using var sr = new StreamReader(path);
            string baseDir = Path.GetDirectoryName(Path.GetFullPath(path)) ?? "";
            Console.WriteLine("[OBJ] Base directory: {0}", baseDir);

            MeshHost mesh = new MeshHost();

            var tempPositions = new List<Float3>();
            var tempTex = new List<Float2>();
            var faceV = new List<int>();
            var faceT = new List<int>();

            string? line;
            string? mtlLibPath = null;
            var currentMtl = -1;
            var mtlNameToIndex = new Dictionary<string, int>(StringComparer.Ordinal);

            var texPathToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            int faceLineCount = 0;

            while ((line = sr.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#') continue;

                if (line.StartsWith("v "))
                {
                    var s = line.AsSpan(2).Trim(); Parse3(s, out float x, out float y, out float z);
                    tempPositions.Add(new Float3(x * scale, y * scale, z * scale));
                }
                else if (line.StartsWith("vt "))
                {
                    var s = line.AsSpan(3).Trim(); Parse2(s, out float u, out float v);
                    tempTex.Add(new Float2(u, v));
                }
                else if (line.StartsWith("f "))
                {
                    faceLineCount++;
                    faceV.Clear(); faceT.Clear();
                    var s = line.AsSpan(2).Trim();
                    int i = 0;
                    while (i < s.Length)
                    {
                        while (i < s.Length && s[i] == ' ') i++;
                        if (i >= s.Length) break;
                        int j = i; while (j < s.Length && s[j] != ' ') j++;
                        var tok = s.Slice(i, j - i);
                        if (tok.Length > 0)
                        {
                            ParseFaceVVT(tok, tempPositions.Count, tempTex.Count, out int vIdx, out int tIdx);
                            faceV.Add(vIdx); faceT.Add(tIdx);
                        }
                        i = j + 1;
                    }

                    if (faceV.Count >= 3)
                    {
                        for (int k = 1; k + 1 < faceV.Count; k++)
                        {
                            if (!flipWinding)
                            {
                                mesh.Triangles.Add(new MeshTri { i0 = faceV[0], i1 = faceV[k], i2 = faceV[k + 1] });
                                mesh.TriUVs.Add(new MeshTriUV { t0 = faceT[0], t1 = faceT[k], t2 = faceT[k + 1] });
                            }
                            else
                            {
                                mesh.Triangles.Add(new MeshTri { i0 = faceV[0], i1 = faceV[k + 1], i2 = faceV[k] });
                                mesh.TriUVs.Add(new MeshTriUV { t0 = faceT[0], t1 = faceT[k + 1], t2 = faceT[k] });
                            }
                            mesh.TriMaterialIndex.Add(currentMtl < 0 ? 0 : currentMtl);
                        }
                    }
                }
                else if (line.StartsWith("mtllib "))
                {
                    var rel = line.Substring(7).Trim();
                    if (!string.IsNullOrWhiteSpace(rel))
                    {
                        mtlLibPath = Path.Combine(baseDir, rel);
                        Console.WriteLine("[OBJ] mtllib: '{0}' -> '{1}' (exists={2})", rel, mtlLibPath, File.Exists(mtlLibPath));
                    }
                }
                else if (line.StartsWith("usemtl "))
                {
                    string name = line.Substring(7).Trim();
                    if (!string.IsNullOrEmpty(name))
                    {
                        if (!mtlNameToIndex.TryGetValue(name, out currentMtl))
                        {
                            currentMtl = mesh.Materials.Count;
                            mtlNameToIndex[name] = currentMtl;
                            mesh.Materials.Add(DefaultMaterial());
                            Console.WriteLine("[OBJ] usemtl new  '{0}' -> idx {1}", name, currentMtl);
                        }
                        else
                        {
                            Console.WriteLine("[OBJ] usemtl bind '{0}' -> idx {1}", name, currentMtl);
                        }
                    }
                }
            }

            Console.WriteLine("[OBJ] Parsed: vertices={0}, texcoords={1}, faces(lines)={2}, triangles={3}",
                tempPositions.Count, tempTex.Count, faceLineCount, mesh.Triangles.Count);

            mesh.Positions.AddRange(tempPositions);
            mesh.Texcoords.AddRange(tempTex);

            // Load & merge MTL materials
            var materialTexPath = new Dictionary<int, string>();
            var alphaTexPath = new Dictionary<int, string>();
            if (!string.IsNullOrWhiteSpace(mtlLibPath) && File.Exists(mtlLibPath))
            {
                var loadedMtls = LoadMtl(mtlLibPath, baseDir, out var diffusePathMap, out var alphaPathMap);
                foreach (var kv in loadedMtls)
                {
                    if (!mtlNameToIndex.TryGetValue(kv.Key, out int idx))
                    {
                        idx = mesh.Materials.Count;
                        mtlNameToIndex[kv.Key] = idx;
                        mesh.Materials.Add(kv.Value);
                    }
                    else
                    {
                        mesh.Materials[idx] = kv.Value;
                    }
                }
                foreach (var kv in diffusePathMap)
                    if (mtlNameToIndex.TryGetValue(kv.Key, out int mi)) materialTexPath[mi] = kv.Value;
                foreach (var kv in alphaPathMap)
                    if (mtlNameToIndex.TryGetValue(kv.Key, out int mi)) alphaTexPath[mi] = kv.Value;
            }

            // Upload textures, remap to global indices (NO fallbacks)
            texPathToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            // Diffuse maps
            foreach (var kv in materialTexPath)
            {
                int matIndex = kv.Key; string p = kv.Value;
                Console.WriteLine("[MTL] map_Kd '{0}' -> idx {1} path '{2}'", GetNameByIndex(mtlNameToIndex, matIndex), matIndex, p);
                if (!texPathToIndex.TryGetValue(p, out int texIndex))
                {
                    if (!TryLoadTextureBGRA(p, out var tex))
                    {
                        var mrSkip = mesh.Materials[matIndex];
                        mrSkip.HasDiffuseMap = 0; mrSkip.DiffuseTexIndex = -1;
                        mesh.Materials[matIndex] = mrSkip;
                        Console.WriteLine("[TEX] MISSING '{0}' – skipping diffuse for material idx {1}", p, matIndex);
                        continue;
                    }
                    texIndex = mesh.Textures.Count;
                    mesh.Textures.Add(tex);
                    texPathToIndex[p] = texIndex;
                    Console.WriteLine("[TEX] -> assigned texture index {0} [{1}x{2}]", texIndex, tex.Width, tex.Height);
                }
                var mr = mesh.Materials[matIndex];
                mr.HasDiffuseMap = 1;
                mr.DiffuseTexIndex = texIndex;
                mesh.Materials[matIndex] = mr;
                Console.WriteLine("[TEX] Material '{0}' (idx {1}) diffuseTexIndex={2}", GetNameByIndex(mtlNameToIndex, matIndex), matIndex, texIndex);
            }

            // Alpha maps (map_d)
            foreach (var kv in alphaTexPath)
            {
                int matIndex = kv.Key; string p = kv.Value;
                Console.WriteLine("[MTL] map_d  '{0}' -> idx {1} path '{2}'", GetNameByIndex(mtlNameToIndex, matIndex), matIndex, p);
                if (!texPathToIndex.TryGetValue(p, out int texIndex))
                {
                    if (!TryLoadTextureBGRA(p, out var tex))
                    {
                        var mrSkip = mesh.Materials[matIndex];
                        mrSkip.HasAlphaMap = 0; mrSkip.AlphaTexIndex = -1; // disable if missing
                        mesh.Materials[matIndex] = mrSkip;
                        Console.WriteLine("[TEX] MISSING '{0}' – skipping alpha map for material idx {1}", p, matIndex);
                        continue;
                    }
                    texIndex = mesh.Textures.Count;
                    mesh.Textures.Add(tex);
                    texPathToIndex[p] = texIndex;
                    Console.WriteLine("[TEX] -> assigned texture index {0} [{1}x{2}]", texIndex, tex.Width, tex.Height);
                }
                var mr = mesh.Materials[matIndex];
                mr.HasAlphaMap = 1; mr.AlphaTexIndex = texIndex;
                mr.TwoSided = 1; // typical for cutouts
                mesh.Materials[matIndex] = mr;
                Console.WriteLine("[TEX] Material '{0}' (idx {1}) alphaTexIndex={2} TwoSided=1", GetNameByIndex(mtlNameToIndex, matIndex), matIndex, texIndex);
            }

            Console.WriteLine("[OBJ] Final counts: Positions={0}, Texcoords={1}, Triangles={2}, Materials={3}, Textures={4}",
                mesh.Positions.Count, mesh.Texcoords.Count, mesh.Triangles.Count, mesh.Materials.Count, mesh.Textures.Count);

            // Dump final materials
            foreach (var kv in mtlNameToIndex)
            {
                int i = kv.Value; var m = mesh.Materials[i];
                Console.WriteLine("[MTL] material '{0}': Kd=({1},{2},{3}) Shading={4} IOR={5} DiffuseMap={6} (idx={7}) AlphaMap={8} (idx={9}) TwoSided={10} AlphaCutoff={11}",
                    kv.Key, m.Kd.X, m.Kd.Y, m.Kd.Z, m.Shading, m.IOR,
                    m.HasDiffuseMap, m.DiffuseTexIndex, m.HasAlphaMap, m.AlphaTexIndex, m.TwoSided, m.AlphaCutoff);
            }

            return mesh;
        }

        private static string GetNameByIndex(Dictionary<string, int> nameToIndex, int idx)
        {
            foreach (var kv in nameToIndex) if (kv.Value == idx) return kv.Key;
            return $"<idx:{idx}>";
        }

        private static MaterialRecord DefaultMaterial() => new MaterialRecord
        {
            Kd = new Float3(0.8f, 0.8f, 0.8f),
            HasDiffuseMap = 0,
            DiffuseTexIndex = -1,
            Shading = MaterialRecord.SHADING_LAMBERT,
            IOR = 1f,
            HasAlphaMap = 0,
            AlphaTexIndex = -1,
            TwoSided = 0,
            AlphaCutoff = 0.5f // sensible default for cutouts
        };

        private static void Parse3(ReadOnlySpan<char> s, out float x, out float y, out float z)
        {
            int i0 = 0; SkipSpaces(s, ref i0);
            int i1 = NextSep(s, i0); x = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
            i0 = i1; SkipSpaces(s, ref i0);
            i1 = NextSep(s, i0); y = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
            i0 = i1; SkipSpaces(s, ref i0);
            i1 = NextSep(s, i0); z = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
        }

        private static void Parse2(ReadOnlySpan<char> s, out float x, out float y)
        {
            int i0 = 0; SkipSpaces(s, ref i0);
            int i1 = NextSep(s, i0); x = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
            i0 = i1; SkipSpaces(s, ref i0);
            i1 = NextSep(s, i0); y = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
        }

        private static void ParseFaceVVT(ReadOnlySpan<char> tok, int vCountSoFar, int tCountSoFar, out int v, out int t)
        {
            int slash1 = tok.IndexOf('/');
            if (slash1 < 0) { v = ParseOneIndex(tok, vCountSoFar); t = 0; return; }
            ReadOnlySpan<char> vspan = tok.Slice(0, slash1);
            v = ParseOneIndex(vspan, vCountSoFar);
            int slash2 = tok.Slice(slash1 + 1).IndexOf('/');
            if (slash2 < 0)
            {
                ReadOnlySpan<char> vtspan = tok.Slice(slash1 + 1);
                t = ParseOneIndex(vtspan, tCountSoFar);
            }
            else
            {
                ReadOnlySpan<char> vtspan = tok.Slice(slash1 + 1, slash2);
                t = vtspan.Length > 0 ? ParseOneIndex(vtspan, tCountSoFar) : 0;
            }
        }

        private static int ParseOneIndex(ReadOnlySpan<char> s, int countSoFar)
        {
            int val = int.Parse(s, CultureInfo.InvariantCulture);
            return val > 0 ? (val - 1) : (countSoFar + val);
        }

        private static void SkipSpaces(ReadOnlySpan<char> s, ref int i) { while (i < s.Length && s[i] == ' ') i++; }
        private static int NextSep(ReadOnlySpan<char> s, int i) { while (i < s.Length && s[i] != ' ') i++; return i; }

        private static Dictionary<string, MaterialRecord> LoadMtl(string mtlPath, string baseDir,
            out Dictionary<string, string> diffuseMapPaths, out Dictionary<string, string> alphaMapPaths)
        {
            Console.WriteLine("[MTL] Loading MTL '{0}'", mtlPath);
            var dict = new Dictionary<string, MaterialRecord>(StringComparer.Ordinal);
            diffuseMapPaths = new Dictionary<string, string>(StringComparer.Ordinal);
            alphaMapPaths = new Dictionary<string, string>(StringComparer.Ordinal);

            string? cur = null;
            MaterialRecord m = DefaultMaterial();

            int mtlCount = 0;
            int kdCount = 0, mapKdCount = 0, mapDCount = 0;

            using var sr = new StreamReader(mtlPath);
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#') continue;

                if (line.StartsWith("newmtl "))
                {
                    if (cur != null)
                    {
                        dict[cur] = m;
                        Console.WriteLine("[MTL] material '{0}': Kd=({1},{2},{3}) Shading={4} IOR={5} DiffuseMap={6} (idx=-1) AlphaMap={7} (idx=-1) TwoSided={8} AlphaCutoff={9}",
                            cur, m.Kd.X, m.Kd.Y, m.Kd.Z, m.Shading, m.IOR, m.HasDiffuseMap, m.HasAlphaMap, m.TwoSided, m.AlphaCutoff);
                    }
                    cur = line.Substring(7).Trim();
                    Console.WriteLine("[MTL] newmtl '{0}'", cur);
                    m = DefaultMaterial();
                    mtlCount++;
                }
                else if (line.StartsWith("Kd "))
                {
                    var s = line.AsSpan(3).Trim();
                    Parse3(s, out float r, out float g, out float b);
                    m.Kd = new Float3(r, g, b);
                    kdCount++;
                    Console.WriteLine("[MTL]   Kd {0} {1} {2}", r, g, b);
                }
                else if (line.StartsWith("map_Kd "))
                {
                    string raw = line.Substring(7).Trim();
                    if (cur != null) diffuseMapPaths[cur] = Path.Combine(baseDir, raw);
                    m.HasDiffuseMap = 1; // actual index resolved later
                    mapKdCount++;
                    Console.WriteLine("[MTL]   map_Kd '{0}' -> '{1}'", raw, Path.Combine(baseDir, raw));
                }
                else if (line.StartsWith("map_d "))
                {
                    string raw = line.Substring(6).Trim();
                    if (cur != null) alphaMapPaths[cur] = Path.Combine(baseDir, raw);
                    m.HasAlphaMap = 1; // index bound later
                    m.TwoSided = 1;    // cutout surfaces are typically two-sided
                    mapDCount++;
                    Console.WriteLine("[MTL]   map_d  '{0}' -> '{1}' (TwoSided=1)", raw, Path.Combine(baseDir, raw));
                }
                else if (line.StartsWith("d "))
                {
                    var s = line.AsSpan(2).Trim();
                    float d = float.Parse(s, CultureInfo.InvariantCulture);
                    Console.WriteLine("[MTL]   d {0} -> TwoSided={1} AlphaCutoff={2}", d, d < 0.999f ? 1 : 0, 0.5f);
                    if (d < 0.999f) { m.TwoSided = 1; m.AlphaCutoff = 0.5f; }
                }
                else if (line.StartsWith("Tr "))
                {
                    var s = line.AsSpan(3).Trim();
                    float tr = float.Parse(s, CultureInfo.InvariantCulture);
                    float d = 1f - tr;
                    Console.WriteLine("[MTL]   Tr {0} (d={1}) -> TwoSided={2} AlphaCutoff={3}", tr, d, d < 0.999f ? 1 : 0, 0.5f);
                    if (d < 0.999f) { m.TwoSided = 1; m.AlphaCutoff = 0.5f; }
                }
                else if (line.StartsWith("Ni "))
                {
                    var s = line.AsSpan(3).Trim();
                    int i0 = 0; SkipSpaces(s, ref i0);
                    int i1 = NextSep(s, i0);
                    m.IOR = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
                    if (m.IOR <= 0f) m.IOR = 1f;
                    Console.WriteLine("[MTL]   Ni {0}", m.IOR);
                }
                else if (line.StartsWith("illum "))
                {
                    int model = int.Parse(line.AsSpan(6).Trim(), CultureInfo.InvariantCulture);
                    if (model >= 5) m.Shading = MaterialRecord.SHADING_GLASS;
                    else if (model >= 3) m.Shading = MaterialRecord.SHADING_MIRROR;
                    else m.Shading = MaterialRecord.SHADING_LAMBERT;
                    Console.WriteLine("[MTL]   illum {0} -> Shading={1}", model, m.Shading);
                }
            }
            if (cur != null)
            {
                dict[cur] = m;
                Console.WriteLine("[MTL] material '{0}': Kd=({1},{2},{3}) Shading={4} IOR={5} DiffuseMap={6} (idx=-1) AlphaMap={7} (idx=-1) TwoSided={8} AlphaCutoff={9}",
                    cur, m.Kd.X, m.Kd.Y, m.Kd.Z, m.Shading, m.IOR, m.HasDiffuseMap, m.HasAlphaMap, m.TwoSided, m.AlphaCutoff);
            }

            Console.WriteLine("[MTL] Loaded materials: {0}", mtlCount);
            Console.WriteLine("[MTL] Read {0} materials, map_Kd={1}, map_d={2}", dict.Count, mapKdCount, mapDCount);
            return dict;
        }

        private static bool TryLoadTextureBGRA(string file, out TextureSrc tex)
        {
            if (!File.Exists(file))
            {
                Console.WriteLine("[TEX] MISSING '{0}' – skipping", file);
                tex = default;
                return false;
            }
            tex = LoadTextureBGRA(file); // assumes file exists
            return true;
        }

        private static TextureSrc LoadTextureBGRA(string file)
        {
            string ext = Path.GetExtension(file).ToLowerInvariant();
            if (ext == ".tga")
            {
                Console.WriteLine("[TEX] TGA '{0}'", file);
                return LoadTgaBGRA(file);
            }

            using var bmp = new Bitmap(file);
            // If not already 32bpp ARGB, convert by direct blit (no blending)
            if (bmp.PixelFormat != PixelFormat.Format32bppArgb)
            {
                using var converted = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(converted))
                {
                    g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                    g.DrawImage(bmp, 0, 0, bmp.Width, bmp.Height);
                }
                return ExtractBitmapBGRA(converted, file);
            }
            return ExtractBitmapBGRA(bmp, file);
        }

        private static TextureSrc ExtractBitmapBGRA(Bitmap bmp, string file)
        {
            int w = bmp.Width, h = bmp.Height;
            var data = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var bd = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    byte* src = (byte*)bd.Scan0.ToPointer();
                    int srcStride = bd.Stride;
                    int dst = 0;
                    for (int y = 0; y < h; y++)
                    {
                        int row = y * srcStride;
                        for (int x = 0; x < w; x++)
                        {
                            int s = row + (x << 2);
                            // Memory layout is BGRA in 32bppArgb
                            data[dst + 0] = src[s + 0];
                            data[dst + 1] = src[s + 1];
                            data[dst + 2] = src[s + 2];
                            data[dst + 3] = src[s + 3];
                            dst += 4;
                        }
                    }
                }
            }
            finally { bmp.UnlockBits(bd); }
            return new TextureSrc { Path = file, Width = w, Height = h, BGRA = data };
        }

        private static TextureSrc LoadTgaBGRA(string file)
        {
            using var fs = File.OpenRead(file);
            using var br = new BinaryReader(fs);
            byte idLength = br.ReadByte();
            byte colorMapType = br.ReadByte();
            byte imageType = br.ReadByte();
            br.ReadUInt16(); br.ReadUInt16(); br.ReadByte(); // CM spec
            br.ReadUInt16(); br.ReadUInt16(); // origin
            ushort width = br.ReadUInt16();
            ushort height = br.ReadUInt16();
            byte pixelDepth = br.ReadByte();
            byte imageDesc = br.ReadByte();
            if (idLength > 0) br.ReadBytes(idLength);

            if (colorMapType != 0) throw new InvalidDataException($"TGA colorMapType={colorMapType} not supported: {file}");

            bool topOrigin = (imageDesc & 0x20) != 0;
            int bpp = pixelDepth == 32 ? 4 : (pixelDepth == 24 ? 3 : (pixelDepth == 8 ? 1 : 0));
            if (bpp == 0) throw new InvalidDataException($"TGA pixelDepth={pixelDepth} not supported: {file}");

            int w = width, h = height;
            Console.WriteLine("[TEX/TGA] '{0}': {1}x{2} {3}bpp imageType={4} topOrigin={5}", file, w, h, bpp * 8, imageType, topOrigin);
            var outData = new byte[w * h * 4];

            void writePixel(int i, byte b, byte g, byte r, byte a)
            {
                int px = i % w, py = i / w;
                int yOut = topOrigin ? py : (h - 1 - py);
                int dst = (yOut * w + px) * 4;
                outData[dst + 0] = b;
                outData[dst + 1] = g;
                outData[dst + 2] = r;
                outData[dst + 3] = a;
            }

            if (imageType == 2 || imageType == 3) // uncompressed
            {
                int total = w * h;
                for (int i = 0; i < total; i++)
                {
                    byte b, g, r, a;
                    if (bpp == 4) { b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = br.ReadByte(); }
                    else if (bpp == 3) { b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = 255; }
                    else { byte y8 = br.ReadByte(); b = g = r = y8; a = 255; }
                    writePixel(i, b, g, r, a);
                }
            }
            else if (imageType == 10) // RLE
            {
                int total = w * h, i = 0;
                while (i < total)
                {
                    byte packet = br.ReadByte();
                    int count = (packet & 0x7F) + 1;
                    if ((packet & 0x80) != 0)
                    {
                        byte b, g, r, a;
                        if (bpp == 4) { b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = br.ReadByte(); }
                        else if (bpp == 3) { b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = 255; }
                        else { byte y8 = br.ReadByte(); b = g = r = y8; a = 255; }
                        for (int k = 0; k < count && i < total; k++, i++) writePixel(i, b, g, r, a);
                    }
                    else
                    {
                        for (int k = 0; k < count && i < total; k++, i++)
                        {
                            byte b, g, r, a;
                            if (bpp == 4) { b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = br.ReadByte(); }
                            else if (bpp == 3) { b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = 255; }
                            else { byte y8 = br.ReadByte(); b = g = r = y8; a = 255; }
                            writePixel(i, b, g, r, a);
                        }
                    }
                }
            }
            else
            {
                throw new InvalidDataException($"TGA imageType={imageType} not supported: {file}");
            }

            return new TextureSrc { Path = file, Width = w, Height = h, BGRA = outData };
        }
    }
}
