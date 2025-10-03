// ==============================
// Engine/RTWindow.cs (zero-copy; single-thread; single PBO)
// ==============================
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using ILGPU.Runtime.Cuda;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

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
        private int _uTexLoc;

        private RTRenderer _renderer;

        private struct PboSlot { public CudaGlInteropIndexBuffer Buf; }
        private PboSlot[] _pbos = Array.Empty<PboSlot>();
        private const int PboCount = 1;

        private int _rtWidth, _rtHeight;

        private bool _mouseCaptured;
        private Vector2 _mousePos, _mouseDeltaAccum, _lastMousePos;

        private readonly string _baseTitle;
        private readonly Stopwatch _perfSw = new();
        private readonly System.Collections.Generic.Queue<(double t, double dt)> _frameSamples = new();
        private double _lastTitleUpdateT = 0.0;
        private const double FrameTimeWindowSec = 5.0;
        private const double FpsWindowSec = 30.0;
        private const double TitleUpdateHz = 4.0;

        private bool _isClosing;
        private int _frameIndex;

        private const int MaxRayPixels = 1_000_000;
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
            _uTexLoc = GL.GetUniformLocation(_program, "uTex");

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
            int aUV  = GL.GetAttribLocation(_program, "aUV");
            GL.EnableVertexAttribArray(aPos);
            GL.VertexAttribPointer(aPos, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), 0);
            GL.EnableVertexAttribArray(aUV);
            GL.VertexAttribPointer(aUV, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), 2 * sizeof(float));
            GL.BindVertexArray(0);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);

            (_rtWidth, _rtHeight) = ComputeInternalRT(_initialWidth, _initialHeight);
            _texture = CreateTexture(_rtWidth, _rtHeight);

            _renderer = new RTRenderer(this, 0);

            CreatePbos(_rtWidth, _rtHeight);

            _mouseCaptured = false;
            _mousePos = Vector2.Zero;
            _mouseDeltaAccum = Vector2.Zero;
            _lastMousePos = MouseState.Position;

            _isClosing = false;
            Title = _baseTitle;
            _perfSw.Start();
            _frameIndex = 0;
            SetMouseCapture(false);
        }

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

            // Simple pacing: serialize GL consumption vs CUDA production
            GL.Finish();

            // Zero-copy render → PBO
            _renderer.RenderDirectToPbo(_pbos[0].Buf, _rtWidth, _rtHeight, _frameIndex, dt);

            // Upload PBO to texture
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, _texture);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, _pbos[0].Buf.glBufferHandle);
            GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, _rtWidth, _rtHeight, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

            // HUD
            double now = _perfSw.Elapsed.TotalSeconds;
            _frameSamples.Enqueue((now, e.Time));
            while (_frameSamples.Count > 0 && _frameSamples.Peek().t < now - FpsWindowSec)
                _frameSamples.Dequeue();

            double sum5 = 0.0; int n5 = 0;
            foreach (var s in _frameSamples) if (s.t >= now - FrameTimeWindowSec) { sum5 += s.dt; n5++; }
            double avgMs5 = n5 > 0 ? (sum5 / n5) * 1000.0 : 0.0;

            int n30 = _frameSamples.Count;
            double span30 = n30 > 0 ? Math.Max(1e-6, now - _frameSamples.Peek().t) : 0.0;
            double fps30 = n30 > 0 ? n30 / span30 : 0.0;

            if (now - _lastTitleUpdateT >= (1.0 / TitleUpdateHz))
            {
                Title = $"{_baseTitle}   |   RT: {_rtWidth}x{_rtHeight}   |   5s avg: {avgMs5.ToString("F2", CultureInfo.InvariantCulture)} ms   |   30s avg: {fps30.ToString("F1", CultureInfo.InvariantCulture)} FPS";
                _lastTitleUpdateT = now;
            }

            // Blit
            GL.Viewport(0, 0, Size.X, Size.Y);
            GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, _texture);
            GL.UseProgram(_program);
            GL.Uniform1(_uTexLoc, 0);
            GL.BindVertexArray(_vao);
            GL.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);
            GL.BindVertexArray(0);
            GL.UseProgram(0);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            SwapBuffers();
            _frameIndex++;
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);

            (int newRtW, int newRtH) = ComputeInternalRT(Math.Max(1, Size.X), Math.Max(1, Size.Y));
            if (newRtW == _rtWidth && newRtH == _rtHeight) return;

            _rtWidth = Math.Max(MinRtDim, newRtW);
            _rtHeight = Math.Max(MinRtDim, newRtH);

            if (_texture != 0) { GL.DeleteTexture(_texture); _texture = 0; }
            _texture = CreateTexture(_rtWidth, _rtHeight);

            DisposePbos();
            CreatePbos(_rtWidth, _rtHeight);
            _frameIndex = 0;
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            _isClosing = true;
            base.OnClosing(e);
        }

        protected override void OnUnload()
        {
            _isClosing = true;

            try { GL.Finish(); } catch { }

            DisposePbos();

            _renderer?.Dispose(); _renderer = null;

            if (_program != 0) { GL.DeleteProgram(_program); _program = 0; }
            if (_vbo != 0) { GL.DeleteBuffer(_vbo); _vbo = 0; }
            if (_vao != 0) { GL.DeleteVertexArray(_vao); _vao = 0; }
            if (_texture != 0) { GL.DeleteTexture(_texture); _texture = 0; }

            base.OnUnload();
        }

        // Input helpers (unchanged)
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
            if (e.Unicode <= char.MaxValue)
                CharTyped?.Invoke((char)e.Unicode);
        }
        protected override void OnMouseMove(MouseMoveEventArgs e)
        { 
            base.OnMouseMove(e); _mousePos = e.Position;
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
            if (!e.IsFocused && _mouseCaptured)
                SetMouseCapture(false);
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
            var delta = _mouseCaptured ? _mouseDeltaAccum : Vector2.Zero;
            _mouseDeltaAccum = Vector2.Zero;
            return new Vector2(-delta.X, delta.Y);
        }

        private void CreatePbos(int w, int h)
        {
            _pbos = new PboSlot[PboCount];
            for (int i = 0; i < PboCount; i++)
                _pbos[i].Buf = new CudaGlInteropIndexBuffer(w * h, _renderer.Accelerator);
        }

        private void DisposePbos()
        {
            if (_pbos == null) return;
            for (int i = 0; i < _pbos.Length; i++)
            {
                try { _pbos[i].Buf?.Dispose(); } catch { }
                _pbos[i].Buf = null;
            }
            _pbos = Array.Empty<PboSlot>();
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
