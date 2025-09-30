// ==============================
// File: Engine/MeshLoaderOBJ.cs
// Replaces your existing MeshLoaderOBJ.cs (adds vt/mtl/usemtl + MTL/texture loading, and TGA support)
// ==============================
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using ILGPU.Algorithms;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
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

    public struct Float2
    {
        public float X;
        public float Y;
        public Float2(float x, float y) { X = x; Y = y; }
    }

    public struct MeshTriUV
    {
        public int t0;
        public int t1;
        public int t2;
    }

    public struct TextureSrc
    {
        public string Path;
        public int Width;
        public int Height;
        public byte[] BGRA;
    }

    public struct MaterialRecord
    {
        public Float3 Kd;
        public int HasDiffuseMap;
        public int DiffuseTexIndex;
        public int _pad;
    }

    public static class MeshLoaderOBJ
    {
        public static MeshHost Load(string path, float scale = 1f, bool flipWinding = true)
        {
            using var sr = new StreamReader(path);
            string baseDir = Path.GetDirectoryName(Path.GetFullPath(path)) ?? "";
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

            while ((line = sr.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#') continue;

                if (line.StartsWith("v "))
                {
                    var s = line.AsSpan(2).Trim();
                    Parse3(s, out float x, out float y, out float z);
                    tempPositions.Add(new Float3(x * scale, y * scale, z * scale));
                }
                else if (line.StartsWith("vt "))
                {
                    var s = line.AsSpan(3).Trim();
                    Parse2(s, out float u, out float v);
                    tempTex.Add(new Float2(u, v));
                }
                else if (line.StartsWith("f "))
                {
                    faceV.Clear();
                    faceT.Clear();
                    var s = line.AsSpan(2).Trim();

                    int i = 0;
                    while (i < s.Length)
                    {
                        while (i < s.Length && s[i] == ' ') i++;
                        if (i >= s.Length) break;
                        int j = i;
                        while (j < s.Length && s[j] != ' ') j++;
                        var tok = s.Slice(i, j - i);
                        if (tok.Length > 0)
                        {
                            ParseFaceVVT(tok, tempPositions.Count, tempTex.Count, out int vIdx, out int tIdx);
                            faceV.Add(vIdx);
                            faceT.Add(tIdx);
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
                            mesh.Materials.Add(new MaterialRecord { Kd = new Float3(0.8f, 0.8f, 0.8f), HasDiffuseMap = 0, DiffuseTexIndex = -1, _pad = 0 });
                        }
                    }
                }
            }

            mesh.Positions.AddRange(tempPositions);
            mesh.Texcoords.AddRange(tempTex);

            if (!string.IsNullOrWhiteSpace(mtlLibPath) && File.Exists(mtlLibPath))
            {
                var loadedMtls = LoadMtl(mtlLibPath, baseDir);
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
            }

            for (int i = 0; i < mesh.Materials.Count; i++)
            {
                var m = mesh.Materials[i];
                if (m.HasDiffuseMap != 0 && m.DiffuseTexIndex >= 0)
                {
                }
            }

            var materialTexPath = new Dictionary<int, string>();
            using var sr2 = new StreamReader(path);
            if (!string.IsNullOrWhiteSpace(mtlLibPath) && File.Exists(mtlLibPath))
            {
                var texMap = LoadMtlTexturePaths(mtlLibPath, baseDir);
                foreach (var kv in texMap)
                {
                    if (mtlNameToIndex.TryGetValue(kv.Key, out int matIndex))
                    {
                        materialTexPath[matIndex] = kv.Value;
                    }
                }
            }

            foreach (var kv in materialTexPath)
            {
                string p = kv.Value;
                if (!texPathToIndex.TryGetValue(p, out int texIndex))
                {
                    var tex = LoadTextureBGRA(p);
                    texIndex = mesh.Textures.Count;
                    mesh.Textures.Add(tex);
                    texPathToIndex[p] = texIndex;
                }
                var mr = mesh.Materials[kv.Key];
                mr.HasDiffuseMap = 1;
                mr.DiffuseTexIndex = texIndex;
                mesh.Materials[kv.Key] = mr;
            }

            return mesh;
        }

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
            if (slash1 < 0)
            {
                v = ParseOneIndex(tok, vCountSoFar);
                t = 0;
                return;
            }
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
                if (vtspan.Length > 0) t = ParseOneIndex(vtspan, tCountSoFar); else t = 0;
            }
        }

        private static int ParseOneIndex(ReadOnlySpan<char> s, int countSoFar)
        {
            int val = int.Parse(s, CultureInfo.InvariantCulture);
            return val > 0 ? (val - 1) : (countSoFar + val);
        }

        private static void SkipSpaces(ReadOnlySpan<char> s, ref int i)
        {
            while (i < s.Length && s[i] == ' ') i++;
        }

        private static int NextSep(ReadOnlySpan<char> s, int i)
        {
            while (i < s.Length && s[i] != ' ') i++;
            return i;
        }

        private static Dictionary<string, MaterialRecord> LoadMtl(string mtlPath, string baseDir)
        {
            var dict = new Dictionary<string, MaterialRecord>(StringComparer.Ordinal);
            string? cur = null;
            Float3 kd = new Float3(0.8f, 0.8f, 0.8f);
            int hasTex = 0;
            int texIndex = -1;

            using var sr = new StreamReader(mtlPath);
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#') continue;
                if (line.StartsWith("newmtl "))
                {
                    if (cur != null)
                    {
                        dict[cur] = new MaterialRecord { Kd = kd, HasDiffuseMap = hasTex, DiffuseTexIndex = texIndex, _pad = 0 };
                    }
                    cur = line.Substring(7).Trim();
                    kd = new Float3(0.8f, 0.8f, 0.8f);
                    hasTex = 0;
                    texIndex = -1;
                }
                else if (line.StartsWith("Kd "))
                {
                    var s = line.AsSpan(3).Trim();
                    Parse3(s, out float r, out float g, out float b);
                    kd = new Float3(r, g, b);
                }
                else if (line.StartsWith("map_Kd "))
                {
                    hasTex = 1;
                }
            }
            if (cur != null)
            {
                dict[cur] = new MaterialRecord { Kd = kd, HasDiffuseMap = hasTex, DiffuseTexIndex = texIndex, _pad = 0 };
            }
            return dict;
        }

        private static Dictionary<string, string> LoadMtlTexturePaths(string mtlPath, string baseDir)
        {
            var dict = new Dictionary<string, string>(StringComparer.Ordinal);
            string? cur = null;

            using var sr = new StreamReader(mtlPath);
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#') continue;
                if (line.StartsWith("newmtl "))
                {
                    cur = line.Substring(7).Trim();
                }
                else if (line.StartsWith("map_Kd ") && cur != null)
                {
                    string raw = line.Substring(7).Trim();
                    string full = Path.Combine(baseDir, raw);
                    dict[cur] = full;
                }
            }
            return dict;
        }

        private static TextureSrc LoadTextureBGRA(string file)
        {
            if (!File.Exists(file))
            {
                return new TextureSrc { Path = file, Width = 1, Height = 1, BGRA = new byte[] { 255, 255, 255, 255 } };
            }
            string ext = Path.GetExtension(file).ToLowerInvariant();
            if (ext == ".tga")
            {
                return LoadTgaBGRA(file);
            }
            using var bmp = new Bitmap(file);
            if (bmp.PixelFormat != PixelFormat.Format32bppArgb)
            {
                using var converted = new Bitmap(bmp.Width, bmp.Height, PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(converted)) { g.DrawImageUnscaled(bmp, 0, 0); }
                return ExtractBitmapBGRA(converted, file);
            }
            return ExtractBitmapBGRA(bmp, file);
        }

        private static TextureSrc ExtractBitmapBGRA(Bitmap bmp, string file)
        {
            int w = bmp.Width;
            int h = bmp.Height;
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
                            data[dst + 0] = src[s + 0];
                            data[dst + 1] = src[s + 1];
                            data[dst + 2] = src[s + 2];
                            data[dst + 3] = src[s + 3];
                            dst += 4;
                        }
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bd);
            }
            return new TextureSrc { Path = file, Width = w, Height = h, BGRA = data };
        }

        private static TextureSrc LoadTgaBGRA(string file)
        {
            using var fs = File.OpenRead(file);
            using var br = new BinaryReader(fs);
            byte idLength = br.ReadByte();
            byte colorMapType = br.ReadByte();
            byte imageType = br.ReadByte();
            ushort cmFirst = br.ReadUInt16();
            ushort cmLength = br.ReadUInt16();
            byte cmDepth = br.ReadByte();
            ushort xOrigin = br.ReadUInt16();
            ushort yOrigin = br.ReadUInt16();
            ushort width = br.ReadUInt16();
            ushort height = br.ReadUInt16();
            byte pixelDepth = br.ReadByte();
            byte imageDesc = br.ReadByte();
            if (idLength > 0) { br.ReadBytes(idLength); }
            if (colorMapType != 0) { return new TextureSrc { Path = file, Width = 1, Height = 1, BGRA = new byte[] { 255, 255, 255, 255 } }; }
            bool topOrigin = (imageDesc & 0x20) != 0;
            int bytesPerPixel = pixelDepth == 32 ? 4 : (pixelDepth == 24 ? 3 : (pixelDepth == 8 ? 1 : 0));
            if (bytesPerPixel == 0) { return new TextureSrc { Path = file, Width = 1, Height = 1, BGRA = new byte[] { 255, 0, 255, 255 } }; }
            int w = width;
            int h = height;
            var outData = new byte[w * h * 4];
            if (imageType == 2 || imageType == 3)
            {
                int total = w * h;
                for (int i = 0; i < total; i++)
                {
                    byte b, g, r, a;
                    if (bytesPerPixel == 4)
                    {
                        b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = br.ReadByte();
                    }
                    else if (bytesPerPixel == 3)
                    {
                        b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = 255;
                    }
                    else
                    {
                        byte y8 = br.ReadByte(); b = y8; g = y8; r = y8; a = 255;
                    }
                    int px = i % w;
                    int py = i / w;
                    int yOut = topOrigin ? py : (h - 1 - py);
                    int dst = (yOut * w + px) * 4;
                    outData[dst + 0] = b;
                    outData[dst + 1] = g;
                    outData[dst + 2] = r;
                    outData[dst + 3] = a;
                }
            }
            else if (imageType == 10)
            {
                int total = w * h;
                int i = 0;
                while (i < total)
                {
                    byte packet = br.ReadByte();
                    int count = (packet & 0x7F) + 1;
                    if ((packet & 0x80) != 0)
                    {
                        byte b, g, r, a;
                        if (bytesPerPixel == 4)
                        {
                            b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = br.ReadByte();
                        }
                        else if (bytesPerPixel == 3)
                        {
                            b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = 255;
                        }
                        else
                        {
                            byte y8 = br.ReadByte(); b = y8; g = y8; r = y8; a = 255;
                        }
                        for (int k = 0; k < count && i < total; k++, i++)
                        {
                            int px = i % w;
                            int py = i / w;
                            int yOut = topOrigin ? py : (h - 1 - py);
                            int dst = (yOut * w + px) * 4;
                            outData[dst + 0] = b;
                            outData[dst + 1] = g;
                            outData[dst + 2] = r;
                            outData[dst + 3] = a;
                        }
                    }
                    else
                    {
                        for (int k = 0; k < count && i < total; k++, i++)
                        {
                            byte b, g, r, a;
                            if (bytesPerPixel == 4)
                            {
                                b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = br.ReadByte();
                            }
                            else if (bytesPerPixel == 3)
                            {
                                b = br.ReadByte(); g = br.ReadByte(); r = br.ReadByte(); a = 255;
                            }
                            else
                            {
                                byte y8 = br.ReadByte(); b = y8; g = y8; r = y8; a = 255;
                            }
                            int px = i % w;
                            int py = i / w;
                            int yOut = topOrigin ? py : (h - 1 - py);
                            int dst = (yOut * w + px) * 4;
                            outData[dst + 0] = b;
                            outData[dst + 1] = g;
                            outData[dst + 2] = r;
                            outData[dst + 3] = a;
                        }
                    }
                }
            }
            else
            {
                return new TextureSrc { Path = file, Width = 1, Height = 1, BGRA = new byte[] { 255, 0, 255, 255 } };
            }
            return new TextureSrc { Path = file, Width = w, Height = h, BGRA = outData };
        }
    }
}
