using ILGPU.Algorithms;
using System.Runtime.CompilerServices;

namespace ILGPU_Raytracing.Engine
{
    public struct Ray
    {
        public Float3 origin;
        public Float3 dir;
        public Float3 invDir;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Ray GenerateRay(Camera cam, float u, float v)
        {
            Float3 dir = Float3.Normalize(cam.lowerLeft + cam.horizontal * u + cam.vertical * v - cam.origin);
            return new Ray { origin = cam.origin, dir = dir, invDir = new Float3(1f / (dir.X != 0f ? dir.X : 1e-8f), 1f / (dir.Y != 0f ? dir.Y : 1e-8f), 1f / (dir.Z != 0f ? dir.Z : 1e-8f)) };
        }
    }

    public struct RNG
    {
        private uint state;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RNG Create(uint seed)
        {
            RNG r;
            r.state = (seed == 0u) ? 1u : seed;
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint NextUInt()
        {
            // xorshift32 step (fast core; seed quality handled by MakeSeed32)
            uint x = state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            state = (x != 0u) ? x : 1u;
            return state;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float NextFloat()
        {
            uint u = NextUInt();
            return (u & 0x00FFFFFFu) * (1.0f / 16777216.0f);
        }

        // --------- High-quality seed mixing ---------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint SplitMix32(ulong x)
        {
            // Steele/Lea SplitMix64 constants, folded to 32 bits
            x += 0x9E3779B97F4A7C15UL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
            x ^= (x >> 31);
            return (uint)(x ^ (x >> 32));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint PcgPermute(uint x)
        {
            // PCG XSH-RR-like output permutation (cheap, good avalanching)
            x ^= x >> 16;
            x *= 0x7FEB352Du;
            x ^= x >> 15;
            x *= 0x846CA68Bu;
            x ^= x >> 16;
            return x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Hash32(uint x)
        {
            x ^= x >> 17; x *= 0xED5AD4BBu;
            x ^= x >> 11; x *= 0xAC4C1B51u;
            x ^= x >> 15; x *= 0x31848BABu;
            x ^= x >> 14;
            return x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint MakeSeed32(uint a, uint b, uint c, uint d)
        {
            // Two 64-bit lanes, then SplitMix and PCG-like permutation
            ulong lane0 = ((ulong)a << 32) | b;
            ulong lane1 = ((ulong)c << 32) | d;
            uint s0 = SplitMix32(lane0 ^ 0xD1B54A32D192ED03UL);
            uint s1 = SplitMix32(lane1 ^ 0x94D049BB133111EBUL);
            uint s = PcgPermute(s0 ^ (RotateLeft(s1, 13) + 0x9E3779B1u));
            s |= 1u; // avoid zero
            return s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint RotateLeft(uint v, int r)
        {
            return (v << (r & 31)) | (v >> ((32 - r) & 31));
        }

        // --------- Convenience constructors ---------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RNG CreateFromIndex1D(int index, int width, int height, int frame, uint sample, uint salt, int lockNoise)
        {
            uint x = (uint)(index % Math.Max(1, width));
            uint y = (uint)(index / Math.Max(1, width));
            return CreateFromPixel(new int2((int)x, (int)y), frame, sample, salt, lockNoise);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RNG CreateFromPixel(int2 pixel, int frame, uint sample, uint salt, int lockNoise)
        {
            uint px = (uint)pixel.x;
            uint py = (uint)pixel.y;

            // If locked, frame contribution is zeroed; otherwise include it.
            uint f = (lockNoise != 0) ? 0u : (uint)frame;

            // When lockNoise != 0, also fold its VALUE into the seed to choose a distinct locked stream.
            uint ln = (uint)lockNoise;
            uint lnMix0 = (lockNoise != 0) ? (Hash32(ln) ^ (ln * 0x1B873593u)) : 0u;
            uint lnMix1 = (lockNoise != 0) ? (RotateLeft(ln, 7) * 0x85EBCA6Bu) : 0u;

            // Two 32-bit lanes for each 64-bit half; all components heavily mixed
            uint lane0a = px ^ 0xB5297A4Du;
            uint lane0b = (py * 0x68E31DA4u) ^ (f * 0x9E3779B1u + 0x85EBCA6Bu) ^ lnMix0;
            uint lane1a = (sample ^ 0xC2B2AE35u) + RotateLeft(px, 16);
            uint lane1b = (salt ^ 0x27D4EB2Fu) + RotateLeft(py, 8) ^ lnMix1;

            uint seed = MakeSeed32(lane0a, lane0b, lane1a, lane1b);
            return Create(seed);
        }
    }

    public struct int2
    {
        public int x;
        public int y;
        public int2(int _x, int _y) { x = _x; y = _y; }
    }
}
