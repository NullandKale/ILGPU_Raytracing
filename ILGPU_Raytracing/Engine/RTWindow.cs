
// ==============================
// File: Engine/RTWindow.cs
// ==============================
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel; // CancelEventArgs
using System.Diagnostics;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using ILGPU.Runtime.Cuda;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework; // Keys, MouseButton

namespace ILGPU_Raytracing.Engine
{
    public sealed class RTWindow : GameWindow
    {
        private readonly int _initialWidth;
        private readonly int _initialHeight;

        private int _vao;
        private int _vbo;
        private int _program;
        private int _texture;

        private RTRenderer _renderer;

        private struct PboSlot
        {
            public CudaGlInteropIndexBuffer Buf; // GL PBO registered with CUDA
            public IntPtr Fence;                 // GLsync
        }

        private PboSlot[] _pbos = Array.Empty<PboSlot>();
        private const int PboCount = 2;

        // Internal raytracing resolution
        private int _rtWidth;
        private int _rtHeight;

        // ====== Input ======
        private bool _mouseCaptured;
        private Vector2 _mousePos;
        private Vector2 _mouseDeltaAccum;
        private Vector2 _lastMousePos; // polled last position (prevents event backlog)

        // ====== Perf stats ======
        private readonly string _baseTitle;
        private readonly Stopwatch _perfSw = new();
        private readonly Queue<(double t, double dt)> _frameSamples = new();
        private double _lastTitleUpdateT = 0.0;
        private const double FrameTimeWindowSec = 5.0;
        private const double FpsWindowSec = 30.0;
        private const double TitleUpdateHz = 4.0;

        // ====== Shutdown ======
        private bool _isClosing;

        // ====== Worker ======
        private Thread _workerThread;
        private CancellationTokenSource _workerCts;
        private readonly ConcurrentQueue<int> _freePbos = new(); // ready for CUDA render→device buffer
        private readonly ConcurrentQueue<int> _readySlots = new(); // device buffer slots ready for present (0/1)
        private readonly List<int> _inFlightPbos = new();          // PBOs DMAing to texture (have fences)
        private readonly AutoResetEvent _freeEvent = new(false);
        private readonly AutoResetEvent _readyEvent = new(false);

        // ====== Frame coupling ======
        private readonly SemaphoreSlim _frameBudget = new(0, 3); // render tickets from GL thread (mailbox: cap backlog)

        // ====== Adaptive resolution cap ======
        private const int MaxRayPixels = 2_000_000;
        private const int MinRtDim = 64;

        public event Action<Keys>? KeyPressed;
        public event Action<Keys>? KeyReleased;
        public event Action<char>? CharTyped;

        public event Action<Vector2>? MouseMoved;
        public event Action<Vector2>? MouseDelta;
        public event Action<MouseButton, Vector2>? MousePressed;
        public event Action<MouseButton, Vector2>? MouseReleased;
        public event Action<Vector2>? MouseWheel;

        public RTWindow(int framebufferWidth, int framebufferHeight, string title)
            : base(
                GameWindowSettings.Default,
                new NativeWindowSettings
                {
                    Size = new Vector2i(framebufferWidth, framebufferHeight),
                    Title = title,
                    Flags = ContextFlags.ForwardCompatible
                })
        {
            _initialWidth = framebufferWidth;
            _initialHeight = framebufferHeight;
            _baseTitle = title;
        }

        protected override void OnLoad()
        {
            base.OnLoad();

            GL.Disable(EnableCap.DepthTest);
            GL.Disable(EnableCap.CullFace);
            GL.ClearColor(0f, 0f, 0f, 1f);

            VSync = VSyncMode.On;

            _program = CreateProgram(VertexSource(), FragmentSource());

            _vao = GL.GenVertexArray();
            _vbo = GL.GenBuffer();
            GL.BindVertexArray(_vao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);

            float[] quad =
            {
                -1f, -1f, 0f, 0f,
                 1f, -1f, 1f, 0f,
                -1f,  1f, 0f, 1f,
                 1f,  1f, 1f, 1f
            };
            GL.BufferData(BufferTarget.ArrayBuffer, quad.Length * sizeof(float), quad, BufferUsageHint.StaticDraw);

            int aPos = GL.GetAttribLocation(_program, "aPos");
            int aUV = GL.GetAttribLocation(_program, "aUV");
            GL.EnableVertexAttribArray(aPos);
            GL.VertexAttribPointer(aPos, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), 0);
            GL.EnableVertexAttribArray(aUV);
            GL.VertexAttribPointer(aUV, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), 2 * sizeof(float));
            GL.BindVertexArray(0);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);

            // Internal RT size
            (_rtWidth, _rtHeight) = ComputeInternalRT(_initialWidth, _initialHeight);
            _texture = CreateTexture(_rtWidth, _rtHeight);

            _renderer = new RTRenderer(this, 0);

            // Create interop PBOs (GL thread)
            CreatePbos(_rtWidth, _rtHeight);
            EnqueueAllPbosAsFree();

            // Input state
            _mouseCaptured = false;
            _mousePos = Vector2.Zero;
            _mouseDeltaAccum = Vector2.Zero;
            _lastMousePos = MouseState.Position;

            _isClosing = false;
            Title = _baseTitle;
            _perfSw.Start();
            SetMouseCapture(false);

            // Start worker
            StartWorker();
        }

        // ---------------- Worker management ----------------

        private void StartWorker()
        {
            _workerCts = new CancellationTokenSource();
            _workerThread = new Thread(WorkerLoop) { IsBackground = true, Name = "CUDA-Renderer" };
            _workerThread.Start(_workerCts.Token);
        }

        private void StopWorker()
        {
            if (_workerThread == null) return;
            try
            {
                _workerCts?.Cancel();
                _readyEvent.Set();
                _freeEvent.Set();
                _frameBudget.Release(3);
                _workerThread.Join();
            }
            catch { }
            finally
            {
                _workerThread = null;
                _workerCts?.Dispose();
                _workerCts = null;
            }
        }

        private void WorkerLoop(object? arg)
        {
            var ct = (CancellationToken)arg!;
            int frame = 0;

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    if (!_frameBudget.Wait(5, ct)) continue;
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                if (!_freePbos.TryDequeue(out int slot))
                {
                    WaitHandle.WaitAny(new[] { _freeEvent, ct.WaitHandle }, 5);
                    if (ct.IsCancellationRequested) break;
                    if (!_freePbos.TryDequeue(out slot))
                        continue;
                }

                int w = _rtWidth;
                int h = _rtHeight;

                try
                {
                    _renderer.RenderToDeviceBuffer(slot, w, h, frame++);
                }
                catch
                {
                    // If resize/shutdown raced, ignore and continue.
                }

                _readySlots.Enqueue(slot);
                _readyEvent.Set();

                Thread.Yield();
            }
        }

        // ---------------- GL thread (input + present) ----------------

        protected override void OnUpdateFrame(FrameEventArgs args)
        {
            base.OnUpdateFrame(args);

            Vector2 polled = MouseState.Position;
            Vector2 delta = polled - _lastMousePos;
            if (!_mouseCaptured) delta = Vector2.Zero;
            if (delta.X != 0f || delta.Y != 0f)
            {
                _mouseDeltaAccum += delta;
                MouseMoved?.Invoke(polled);
                MouseDelta?.Invoke(delta);
            }
            _mousePos = polled;
            _lastMousePos = polled;
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);
            if (_isClosing) return;

            float dt = (float)Math.Clamp(e.Time, 0.0, 0.25);
            _renderer.UpdateCamera(dt);

            // Hand out at most one render "ticket" per GL frame (cap backlog to ~2 tickets total).
            if (_frameBudget.CurrentCount < 2) _frameBudget.Release();

            int toPresentSlot = -1;
            while (_readySlots.TryDequeue(out int s))
            {
                if (toPresentSlot != -1)
                {
                    _freePbos.Enqueue(s);
                    _freeEvent.Set();
                }
                toPresentSlot = s;
            }

            if (toPresentSlot != -1)
            {
                int pboIndex = toPresentSlot;

                _renderer.CopyDeviceToPbo(toPresentSlot, _pbos[pboIndex].Buf, _rtWidth, _rtHeight);

                GL.ActiveTexture(TextureUnit.Texture0);
                GL.BindTexture(TextureTarget.Texture2D, _texture);
                GL.BindBuffer(BufferTarget.PixelUnpackBuffer, _pbos[pboIndex].Buf.glBufferHandle);
                GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
                GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, _rtWidth, _rtHeight, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
                GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

                IntPtr fence = GL.FenceSync(SyncCondition.SyncGpuCommandsComplete, WaitSyncFlags.None);
                _pbos[pboIndex].Fence = fence;
                _inFlightPbos.Add(pboIndex);
            }

            for (int i = _inFlightPbos.Count - 1; i >= 0; --i)
            {
                int idx = _inFlightPbos[i];
                var f = _pbos[idx].Fence;
                if (f != IntPtr.Zero)
                {
                    var res = GL.ClientWaitSync(f, ClientWaitSyncFlags.SyncFlushCommandsBit, 0);
                    if (res == WaitSyncStatus.AlreadySignaled || res == WaitSyncStatus.ConditionSatisfied)
                    {
                        GL.DeleteSync(f);
                        _pbos[idx].Fence = IntPtr.Zero;
                        _inFlightPbos.RemoveAt(i);
                        _freePbos.Enqueue(idx);
                        _freeEvent.Set();
                    }
                }
            }

            double now = _perfSw.Elapsed.TotalSeconds;
            _frameSamples.Enqueue((now, e.Time));
            while (_frameSamples.Count > 0 && _frameSamples.Peek().t < now - FpsWindowSec)
                _frameSamples.Dequeue();

            double sum5 = 0.0; int n5 = 0;
            foreach (var s in _frameSamples)
                if (s.t >= now - FrameTimeWindowSec) { sum5 += s.dt; n5++; }
            double avgMs5 = n5 > 0 ? (sum5 / n5) * 1000.0 : 0.0;

            int n30 = _frameSamples.Count;
            double span30 = n30 > 0 ? Math.Max(1e-6, now - _frameSamples.Peek().t) : 0.0;
            double fps30 = n30 > 0 ? n30 / span30 : 0.0;

            if (now - _lastTitleUpdateT >= (1.0 / TitleUpdateHz))
            {
                Title = $"{_baseTitle}   |   RT: {_rtWidth}x{_rtHeight}   |   5s avg: {avgMs5.ToString("F2", CultureInfo.InvariantCulture)} ms   |   30s avg: {fps30.ToString("F1", CultureInfo.InvariantCulture)} FPS";
                _lastTitleUpdateT = now;
            }

            GL.Viewport(0, 0, Size.X, Size.Y);
            GL.Clear(ClearBufferMask.ColorBufferBit);

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, _texture);
            GL.UseProgram(_program);
            int texLoc = GL.GetUniformLocation(_program, "uTex");
            GL.Uniform1(texLoc, 0);
            GL.BindVertexArray(_vao);
            GL.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);
            GL.BindVertexArray(0);
            GL.UseProgram(0);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            SwapBuffers();
        }

        // ---------------- Resize / teardown ----------------

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);

            (int newRtW, int newRtH) = ComputeInternalRT(Math.Max(1, Size.X), Math.Max(1, Size.Y));
            if (newRtW == _rtWidth && newRtH == _rtHeight)
                return;

            StopWorker();
            DrainAndDeleteFences();

            _rtWidth = Math.Max(MinRtDim, newRtW);
            _rtHeight = Math.Max(MinRtDim, newRtH);

            if (_texture != 0) { GL.DeleteTexture(_texture); _texture = 0; }
            _texture = CreateTexture(_rtWidth, _rtHeight);

            DisposePbos();
            CreatePbos(_rtWidth, _rtHeight);
            EnqueueAllPbosAsFree();

            StartWorker();
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            _isClosing = true;
            base.OnClosing(e);
        }

        protected override void OnUnload()
        {
            _isClosing = true;

            StopWorker();
            DrainAndDeleteFences();

            try { GL.Finish(); } catch { }

            DisposePbos();

            _renderer?.Dispose(); _renderer = null;

            if (_program != 0) { GL.DeleteProgram(_program); _program = 0; }
            if (_vbo != 0) { GL.DeleteBuffer(_vbo); _vbo = 0; }
            if (_vao != 0) { GL.DeleteVertexArray(_vao); _vao = 0; }
            if (_texture != 0) { GL.DeleteTexture(_texture); _texture = 0; }

            base.OnUnload();
        }

        // ---------------- Input helpers ----------------

        protected override void OnKeyDown(KeyboardKeyEventArgs e)
        {
            base.OnKeyDown(e);
            if (!e.IsRepeat)
            {
                if (e.Key == Keys.E) SetMouseCapture(!_mouseCaptured);
                KeyPressed?.Invoke(e.Key);
            }
        }
        protected override void OnKeyUp(KeyboardKeyEventArgs e)
        {
            base.OnKeyUp(e);
            KeyReleased?.Invoke(e.Key);
        }
        protected override void OnTextInput(TextInputEventArgs e)
        {
            base.OnTextInput(e);
            if (e.Unicode <= char.MaxValue) CharTyped?.Invoke((char)e.Unicode);
        }
        protected override void OnMouseMove(MouseMoveEventArgs e)
        {
            base.OnMouseMove(e);
            _mousePos = e.Position;
        }
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            MousePressed?.Invoke(e.Button, _mousePos);
        }
        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            base.OnMouseUp(e);
            MouseReleased?.Invoke(e.Button, _mousePos);
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            MouseWheel?.Invoke(e.Offset);
        }
        protected override void OnFocusedChanged(FocusedChangedEventArgs e)
        {
            base.OnFocusedChanged(e);
            if (!e.IsFocused && _mouseCaptured) SetMouseCapture(false);
        }

        public void SetMouseCapture(bool capture)
        {
            _mouseCaptured = capture;
            CursorState = capture ? OpenTK.Windowing.Common.CursorState.Grabbed : OpenTK.Windowing.Common.CursorState.Normal;
            _mouseDeltaAccum = Vector2.Zero;
            _lastMousePos = MouseState.Position;
        }
        public bool IsKeyDown(Keys key) => KeyboardState.IsKeyDown(key);
        public bool IsMouseDown(MouseButton btn) => MouseState.IsButtonDown(btn);
        public Vector2 GetMousePosition() => _mousePos;
        public bool IsMouseCaptured => _mouseCaptured;

        public Vector2 ConsumeMouseDelta()
        {
            Vector2 polled = MouseState.Position;
            Vector2 d = _mouseCaptured ? (polled - _lastMousePos) : Vector2.Zero;
            _lastMousePos = polled;
            var accum = _mouseDeltaAccum;
            _mouseDeltaAccum = Vector2.Zero;
            Vector2 total = d + accum;
            return new Vector2(-total.X, total.Y);
        }

        // ---------------- PBO / GL helpers ----------------

        private void CreatePbos(int w, int h)
        {
            _pbos = new PboSlot[PboCount];
            using (var binding = _renderer.Accelerator.BindScoped())
            {
                for (int i = 0; i < PboCount; i++)
                {
                    _pbos[i].Buf = new CudaGlInteropIndexBuffer(w * h, _renderer.Accelerator);
                    _pbos[i].Fence = IntPtr.Zero;
                }
                binding.Recover();
            }
        }

        private void DisposePbos()
        {
            if (_pbos == null) return;
            for (int i = 0; i < _pbos.Length; i++)
            {
                if (_pbos[i].Fence != IntPtr.Zero)
                {
                    try { GL.DeleteSync(_pbos[i].Fence); } catch { }
                    _pbos[i].Fence = IntPtr.Zero;
                }
                try { _pbos[i].Buf?.Dispose(); } catch { }
                _pbos[i].Buf = null;
            }
            _pbos = Array.Empty<PboSlot>();
            while (_freePbos.TryDequeue(out _)) { }
            while (_readySlots.TryDequeue(out _)) { }
            _inFlightPbos.Clear();
        }

        private void EnqueueAllPbosAsFree()
        {
            for (int i = 0; i < _pbos.Length; i++)
                _freePbos.Enqueue(i);
            _freeEvent.Set();
        }

        private void DrainAndDeleteFences()
        {
            for (int i = 0; i < _pbos.Length; i++)
            {
                var f = _pbos[i].Fence;
                if (f != IntPtr.Zero)
                {
                    try
                    {
                        var res = GL.ClientWaitSync(f, ClientWaitSyncFlags.SyncFlushCommandsBit, 10_000_000); // 10ms
                        if (res == WaitSyncStatus.TimeoutExpired)
                            GL.Finish();
                        GL.DeleteSync(f);
                    }
                    catch { }
                    _pbos[i].Fence = IntPtr.Zero;
                }
            }
            _inFlightPbos.Clear();
        }

        private static (int w, int h) ComputeInternalRT(int winW, int winH)
        {
            long winPix = (long)winW * winH;
            if (winPix <= MaxRayPixels) return (winW, winH);
            double scale = Math.Sqrt((double)MaxRayPixels / (double)winPix);
            int w = Math.Max(MinRtDim, (int)Math.Round(winW * scale));
            int h = Math.Max(MinRtDim, (int)Math.Round(winH * scale));
            return (w, h);
        }

        private static int CreateTexture(int width, int height)
        {
            int tex = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, tex);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, width, height, 0, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            GL.BindTexture(TextureTarget.Texture2D, 0);
            return tex;
        }

        private static int CreateProgram(string vs, string fs)
        {
            int v = GL.CreateShader(ShaderType.VertexShader);
            GL.ShaderSource(v, vs);
            GL.CompileShader(v);
            GL.GetShader(v, ShaderParameter.CompileStatus, out int vStatus);
            if (vStatus != (int)All.True)
            {
                string log = GL.GetShaderInfoLog(v);
                GL.DeleteShader(v);
                throw new InvalidOperationException("Vertex shader compile error: " + log);
            }

            int f = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(f, fs);
            GL.CompileShader(f);
            GL.GetShader(f, ShaderParameter.CompileStatus, out int fStatus);
            if (fStatus != (int)All.True)
            {
                string log = GL.GetShaderInfoLog(f);
                GL.DeleteShader(v);
                GL.DeleteShader(f);
                throw new InvalidOperationException("Fragment shader compile error: " + log);
            }

            int p = GL.CreateProgram();
            GL.AttachShader(p, v);
            GL.AttachShader(p, f);
            GL.LinkProgram(p);
            GL.GetProgram(p, GetProgramParameterName.LinkStatus, out int linkStatus);
            GL.DetachShader(p, v);
            GL.DetachShader(p, f);
            GL.DeleteShader(v);
            GL.DeleteShader(f);
            if (linkStatus != (int)All.True)
            {
                string log = GL.GetProgramInfoLog(p);
                GL.DeleteProgram(p);
                throw new InvalidOperationException("Program link error: " + log);
            }
            return p;
        }

        private static string VertexSource() => @"#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main()
{
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}";

        private static string FragmentSource() => @"#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main()
{
    FragColor = texture(uTex, vUV);
}";
    }
}
