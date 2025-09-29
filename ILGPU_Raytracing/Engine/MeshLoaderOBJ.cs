using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace ILGPU_Raytracing.Engine
{
    public sealed class MeshHost
    {
        public readonly List<Float3> Positions = new List<Float3>();
        public readonly List<MeshTri> Triangles = new List<MeshTri>();
    }

    public static class MeshLoaderOBJ
    {
        // flipWinding lets you quickly try reversing the face order if normals look inverted.
        public static MeshHost Load(string path, float scale = 1f, bool flipWinding = true)
        {
            using var sr = new StreamReader(path);
            MeshHost mesh = new MeshHost();
            var tempPositions = new List<Float3>();
            var faceIdx = new List<int>();
            string? line;

            while ((line = sr.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#') continue;

                if (line.StartsWith("v "))
                {
                    var s = line.AsSpan(2).Trim();
                    Parse3(s, out float x, out float y, out float z);
                    tempPositions.Add(new Float3(x * scale, y * scale, z * scale));
                }
                else if (line.StartsWith("f "))
                {
                    faceIdx.Clear();
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
                            faceIdx.Add(ParseFaceVertex(tok, tempPositions.Count));
                        i = j + 1;
                    }

                    if (faceIdx.Count >= 3)
                    {
                        for (int k = 1; k + 1 < faceIdx.Count; k++)
                        {
                            if (!flipWinding)
                                mesh.Triangles.Add(new MeshTri { i0 = faceIdx[0], i1 = faceIdx[k], i2 = faceIdx[k + 1] });
                            else
                                mesh.Triangles.Add(new MeshTri { i0 = faceIdx[0], i1 = faceIdx[k + 1], i2 = faceIdx[k] });
                        }
                    }
                }
            }

            mesh.Positions.AddRange(tempPositions);
            return mesh;

            static void Parse3(ReadOnlySpan<char> s, out float x, out float y, out float z)
            {
                int i0 = 0; SkipSpaces(s, ref i0);
                int i1 = NextSep(s, i0); x = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
                i0 = i1; SkipSpaces(s, ref i0);
                i1 = NextSep(s, i0); y = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
                i0 = i1; SkipSpaces(s, ref i0);
                i1 = NextSep(s, i0); z = float.Parse(s.Slice(i0, i1 - i0), CultureInfo.InvariantCulture);
            }

            // Handle v, v/vt, v//vn, v/vt/vn and NEGATIVE indices.
            static int ParseFaceVertex(ReadOnlySpan<char> tok, int vertexCountSoFar)
            {
                int slash = tok.IndexOf('/');
                ReadOnlySpan<char> vspan = slash < 0 ? tok : tok.Slice(0, slash);
                int v = int.Parse(vspan, CultureInfo.InvariantCulture);
                return v > 0 ? (v - 1) : (vertexCountSoFar + v);
            }

            static void SkipSpaces(ReadOnlySpan<char> s, ref int i)
            {
                while (i < s.Length && s[i] == ' ') i++;
            }

            static int NextSep(ReadOnlySpan<char> s, int i)
            {
                while (i < s.Length && s[i] != ' ') i++;
                return i;
            }
        }
    }
}
