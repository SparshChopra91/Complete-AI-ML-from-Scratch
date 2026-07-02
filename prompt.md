first of all analyze the all the files in this folder 
then i only want you to crate me the frontend in the app.py 
don't edit any other file you are only required to create the app.py file 
I am building the app.py front-end for a "Smart Clinic Assistant" using Streamlit. The backend consists of a Two-Tier Machine Learning Triage Engine (Random Forest) and a Generative AI clinical translation layer. I need you to act as an expert UI/UX designer and Streamlit developer to write the complete, production-ready app.py script.

1. Global Design System & UI/UX (The "Light Theme Particle" Aesthetic):

Override Streamlit's default styling using st.set_page_config (Wide layout, collapsed sidebar).

Inject custom CSS (st.markdown) to enforce a premium, modern Light Theme. Use clinical, trustworthy colors: crisp whites, soft grays, and modern medical blues (e.g., #0A66C2 or #E8F0FE).

Particle Animation: Inject a lightweight CSS/JS snippet (like particles.js or a pure CSS floating animation) into the background to create a subtle, slow-moving geometric particle effect that looks high-tech but does not distract the user.

Round the corners of all input boxes and buttons (border-radius), add soft drop-shadows to containers, and use a clean sans-serif font (like Inter or Roboto).

2. The Layout Architecture:

Header: A sleek banner at the top with a logo icon (🏥), the title "Smart Clinic Triage Engine," and a subtitle explaining it is an AI-driven, cost-aware diagnostic assistant.

Main Layout: Use st.columns to create a 2-column layout.

Left Column (Data Entry): The intake forms.

Right Column (AI Output): The triage routing decision and GenAI medical notes.

3. The Functional Inputs (Session State & Logic):

Use st.session_state to track if the patient is in "Tier 1" or if they have been pushed to "Tier 2 (Gray Zone)".

Form 1 (Tier 1 Vitals): Create clean, intuitive inputs for Age (number), Sex (dropdown), Resting Blood Pressure (slider), and Chest Pain Type (dropdown). Include a primary "Run Initial Triage" button.

Form 2 (Tier 2 Labs - Conditionally rendered): If the AI determines the patient is in the "Gray Zone", dynamically reveal inputs for Cholesterol, Max Heart Rate, Fasting Blood Sugar, and Resting ECG and other features too see the files. Include a "Run Final Diagnosis" button.

4. The Output Dashboard (Right Column):

The Uncertainty Gate Display: Use colored metric cards (st.metric or custom HTML divs) to display the AI's risk probability. Use color-coding: Green for Low Risk (Discharge), Red for High Risk (Admit), and Yellow/Orange for the Gray Zone (Request Labs).

The LLM Explanation Box: Create a distinct, visually elevated container titled "🤖 AI Clinical Note". This is where the Gemini translation of the SHAP values will be displayed. Style it like a real medical chart note.

5. Code Integration:


Provide the full Python code, perfectly formatted, with comments explaining exactly where the CSS injection is happening.

start cooking 
there are a lot of things to correct in this 
1. use the colour combination of the a soft warm light appeling to eyes colour accorss the website and use the box inside box design that looks clean and professional
2. the major problem after it shows the accuracy the tier 2 model should automatically be launchesd if the person is in danger zone but it doesn't 
3. the fiiling buttoin like male female just shows blank and black colour this happens across the webiste with the logo too 
4. completely rebrand the whole website to make it loook like the ui designa nd ux with the best looks 
5. i have a component from the react bits to be added in the background of the whole websote and the colour combination should be according to it 
6. increase the functionality by showing the shap values graphs pie charts etc etc 
7. create the best user experinece so smoothness as well as sdesign 








react bits 
Side Rays
Preview
Code

Copy Prompt
Install

CLI
Manual
pnpm
npm
yarn
bun
npm install ogl
Usage
import SideRays from './SideRays';

<div style={{ width: '100%', height: '600px', position: 'relative' }}>
  <SideRays
    speed={2.5}
    rayColor1="#EAB308"
    rayColor2="#96c8ff"
    intensity={2}
    spread={2}
    origin="top-right"
    tilt={0}
    saturation={1.5}
    blend={0.75}
    falloff={1.6}
    opacity={1}
  />
</div>
code

JS

CSS
import { useRef, useEffect, useState } from 'react';
import { Renderer, Program, Triangle, Mesh } from 'ogl';
import './SideRays.css';

const hexToRgb = hex => {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255] : [1, 1, 1];
};

const originToFlip = origin => {
  switch (origin) {
    case 'top-left': return [1, 0];
    case 'bottom-right': return [0, 1];
    case 'bottom-left': return [1, 1];
    default: return [0, 0];
  }
};

const SideRays = ({
  speed = 2.5,
  rayColor1 = '#EAB308',
  rayColor2 = '#96c8ff',
  intensity = 2,
  spread = 2,
  origin = 'top-right',
  tilt = 0,
  saturation = 1.5,
  blend = 0.75,
  falloff = 1.6,
  opacity = 1.0,
  className = ''
}) => {
  const containerRef = useRef(null);
  const uniformsRef = useRef(null);
  const rendererRef = useRef(null);
  const animationIdRef = useRef(null);
  const meshRef = useRef(null);
  const cleanupFunctionRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);
  const observerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    observerRef.current = new IntersectionObserver(
      entries => {
        const entry = entries[0];
        setIsVisible(entry.isIntersecting);
      },
      { threshold: 0.1 }
    );

    observerRef.current.observe(containerRef.current);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
        observerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!isVisible || !containerRef.current) return;

    if (cleanupFunctionRef.current) {
      cleanupFunctionRef.current();
      cleanupFunctionRef.current = null;
    }

    const initializeWebGL = async () => {
      if (!containerRef.current) return;

      await new Promise(resolve => setTimeout(resolve, 10));

      if (!containerRef.current) return;

      const renderer = new Renderer({
        dpr: Math.min(window.devicePixelRatio, 2),
        alpha: true
      });
      rendererRef.current = renderer;

      const gl = renderer.gl;
      gl.canvas.style.width = '100%';
      gl.canvas.style.height = '100%';

      while (containerRef.current.firstChild) {
        containerRef.current.removeChild(containerRef.current.firstChild);
      }
      containerRef.current.appendChild(gl.canvas);

      const vert = `
attribute vec2 position;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
}`;

      const frag = `precision highp float;

uniform float iTime;
uniform vec2 iResolution;
uniform float iSpeed;
uniform vec3 iRayColor1;
uniform vec3 iRayColor2;
uniform float iIntensity;
uniform float iSpread;
uniform float iFlipX;
uniform float iFlipY;
uniform float iTilt;
uniform float iSaturation;
uniform float iBlend;
uniform float iFalloff;
uniform float iOpacity;

float rayStrength(vec2 raySource, vec2 rayRefDirection, vec2 coord, float seedA, float seedB, float speed) {
  vec2 sourceToCoord = coord - raySource;
  float cosAngle = dot(normalize(sourceToCoord), rayRefDirection);
  return clamp(
    (0.45 + 0.15 * sin(cosAngle * seedA + iTime * speed)) +
    (0.3 + 0.2 * cos(-cosAngle * seedB + iTime * speed)),
    0.0, 1.0) *
    clamp((iResolution.x - length(sourceToCoord)) / iResolution.x, 0.5, 1.0);
}

void main() {
  vec2 fragCoord = gl_FragCoord.xy;
  if (iFlipX > 0.5) fragCoord.x = iResolution.x - fragCoord.x;
  if (iFlipY > 0.5) fragCoord.y = iResolution.y - fragCoord.y;

  vec2 coord = vec2(fragCoord.x, iResolution.y - fragCoord.y);
  vec2 rayPos = vec2(iResolution.x * 1.1, -0.5 * iResolution.y);

  float tiltRad = iTilt * 3.14159265 / 180.0;
  float cs = cos(tiltRad);
  float sn = sin(tiltRad);
  vec2 rel = coord - rayPos;
  vec2 tiltedCoord = vec2(rel.x * cs - rel.y * sn, rel.x * sn + rel.y * cs) + rayPos;

  float halfSpread = iSpread * 0.275;
  vec2 rayRefDir1 = normalize(vec2(cos(0.785398 + halfSpread), sin(0.785398 + halfSpread)));
  vec2 rayRefDir2 = normalize(vec2(cos(0.785398 - halfSpread), sin(0.785398 - halfSpread)));

  vec4 rays1 = vec4(iRayColor1, 1.0) * rayStrength(rayPos, rayRefDir1, tiltedCoord, 36.2214, 21.11349, iSpeed);
  vec4 rays2 = vec4(iRayColor2, 1.0) * rayStrength(rayPos, rayRefDir2, tiltedCoord, 22.3991, 18.0234, iSpeed * 0.2);

  vec4 color = rays1 * (1.0 - iBlend) * 0.9 + rays2 * iBlend * 0.9;

  float distanceToLight = length(fragCoord.xy - vec2(rayPos.x, iResolution.y - rayPos.y)) / iResolution.y;
  float brightness = iIntensity * 0.4 / pow(max(distanceToLight, 0.001), iFalloff);
  color.rgb *= brightness;

  float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
  color.rgb = mix(vec3(gray), color.rgb, iSaturation);

  color.a = max(color.r, max(color.g, color.b)) * iOpacity;
  gl_FragColor = color;
}`;

      const [flipX, flipY] = originToFlip(origin);
      const uniforms = {
        iTime: { value: 0 },
        iResolution: { value: [1, 1] },
        iSpeed: { value: speed },
        iRayColor1: { value: hexToRgb(rayColor1) },
        iRayColor2: { value: hexToRgb(rayColor2) },
        iIntensity: { value: intensity },
        iSpread: { value: spread },
        iFlipX: { value: flipX },
        iFlipY: { value: flipY },
        iTilt: { value: tilt },
        iSaturation: { value: saturation },
        iBlend: { value: blend },
        iFalloff: { value: falloff },
        iOpacity: { value: opacity }
      };
      uniformsRef.current = uniforms;

      const geometry = new Triangle(gl);
      const program = new Program(gl, { vertex: vert, fragment: frag, uniforms });
      const mesh = new Mesh(gl, { geometry, program });
      meshRef.current = mesh;

      const updateSize = () => {
        if (!containerRef.current || !renderer) return;
        renderer.dpr = Math.min(window.devicePixelRatio, 2);
        const { clientWidth: w, clientHeight: h } = containerRef.current;
        renderer.setSize(w, h);
        uniforms.iResolution.value = [w * renderer.dpr, h * renderer.dpr];
      };

      const loop = t => {
        if (!rendererRef.current || !uniformsRef.current || !meshRef.current) return;
        uniforms.iTime.value = t * 0.001;
        try {
          renderer.render({ scene: mesh });
          animationIdRef.current = requestAnimationFrame(loop);
        } catch (e) {
          return;
        }
      };

      window.addEventListener('resize', updateSize);
      updateSize();
      animationIdRef.current = requestAnimationFrame(loop);

      cleanupFunctionRef.current = () => {
        if (animationIdRef.current) {
          cancelAnimationFrame(animationIdRef.current);
          animationIdRef.current = null;
        }
        window.removeEventListener('resize', updateSize);
        if (renderer) {
          try {
            const loseCtx = renderer.gl.getExtension('WEBGL_lose_context');
            if (loseCtx) loseCtx.loseContext();
            const canvas = renderer.gl.canvas;
            if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
          } catch (e) {}
        }
        rendererRef.current = null;
        uniformsRef.current = null;
        meshRef.current = null;
      };
    };

    initializeWebGL();

    return () => {
      if (cleanupFunctionRef.current) {
        cleanupFunctionRef.current();
        cleanupFunctionRef.current = null;
      }
    };
  }, [isVisible, speed, rayColor1, rayColor2, intensity, spread, origin, tilt, saturation, blend, falloff, opacity]);

  useEffect(() => {
    if (!uniformsRef.current) return;
    const u = uniformsRef.current;
    u.iSpeed.value = speed;
    u.iRayColor1.value = hexToRgb(rayColor1);
    u.iRayColor2.value = hexToRgb(rayColor2);
    u.iIntensity.value = intensity;
    u.iSpread.value = spread;
    const [flipX, flipY] = originToFlip(origin);
    u.iFlipX.value = flipX;
    u.iFlipY.value = flipY;
    u.iTilt.value = tilt;
    u.iSaturation.value = saturation;
    u.iBlend.value = blend;
    u.iFalloff.value = falloff;
    u.iOpacity.value = opacity;
  }, [speed, rayColor1, rayColor2, intensity, spread, origin, tilt, saturation, blend, falloff, opacity]);

  return <div ref={containerRef} className={`side-rays-container ${className}`.trim()} />;
};

export default SideRays;
CSS
.side-rays-container {
  width: 100%;
  height: 100%;
  position: relative;
  pointer-events: none;
  z-index: 3;
  overflow: hidden;
}

