import { useRef, useEffect } from 'react'
import * as THREE from 'three'

interface DNAHelixProps {
  scrollProgress: number // 0–1
}

export default function DNAHelix({ scrollProgress }: DNAHelixProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<{
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    renderer: THREE.WebGLRenderer
    helixGroup: THREE.Group
    particles: THREE.Points
    frameId: number
  } | null>(null)

  useEffect(() => {
    if (!mountRef.current) return

    const width = window.innerWidth
    const height = window.innerHeight

    // Scene
    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x000a0f, 0.015)

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000)
    camera.position.set(0, 0, 30)

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000a0f, 1)
    mountRef.current.appendChild(renderer.domElement)

    // ── DNA Double Helix ──────────────────────────────────────
    const helixGroup = new THREE.Group()
    const helixHeight = 60
    const radius = 3
    const turns = 6
    const pointsPerStrand = 300

    // Matrix-green glow material for strands
    const strandMaterial = new THREE.LineBasicMaterial({
      color: 0x00ff41,
      transparent: true,
      opacity: 0.8,
    })

    // Build two helical strands
    for (let strand = 0; strand < 2; strand++) {
      const points: THREE.Vector3[] = []
      const offset = strand * Math.PI

      for (let i = 0; i < pointsPerStrand; i++) {
        const t = i / pointsPerStrand
        const angle = t * turns * Math.PI * 2 + offset
        const y = (t - 0.5) * helixHeight
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius
        points.push(new THREE.Vector3(x, y, z))
      }

      const geometry = new THREE.BufferGeometry().setFromPoints(points)
      const line = new THREE.Line(geometry, strandMaterial)
      helixGroup.add(line)
    }

    // Base pair rungs connecting the strands
    const rungMaterial = new THREE.LineBasicMaterial({
      color: 0x00cc33,
      transparent: true,
      opacity: 0.3,
    })

    const rungCount = 80
    for (let i = 0; i < rungCount; i++) {
      const t = i / rungCount
      const angle = t * turns * Math.PI * 2
      const y = (t - 0.5) * helixHeight

      const x1 = Math.cos(angle) * radius
      const z1 = Math.sin(angle) * radius
      const x2 = Math.cos(angle + Math.PI) * radius
      const z2 = Math.sin(angle + Math.PI) * radius

      const rungGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x1, y, z1),
        new THREE.Vector3(x2, y, z2),
      ])
      helixGroup.add(new THREE.Line(rungGeom, rungMaterial))
    }

    // Glowing spheres at base pair junctions
    const sphereGeom = new THREE.SphereGeometry(0.12, 8, 8)
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.9,
    })

    for (let strand = 0; strand < 2; strand++) {
      const offset = strand * Math.PI
      for (let i = 0; i < rungCount; i++) {
        const t = i / rungCount
        const angle = t * turns * Math.PI * 2 + offset
        const y = (t - 0.5) * helixHeight
        const x = Math.cos(angle) * radius
        const z = Math.sin(angle) * radius
        const sphere = new THREE.Mesh(sphereGeom, sphereMat)
        sphere.position.set(x, y, z)
        helixGroup.add(sphere)
      }
    }

    scene.add(helixGroup)

    // ── Floating matrix particles ─────────────────────────────
    const particleCount = 2000
    const particlePositions = new Float32Array(particleCount * 3)
    const particleColors = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3] = (Math.random() - 0.5) * 100
      particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 100
      particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 100

      // Green-tinted particles with variation
      const g = 0.5 + Math.random() * 0.5
      particleColors[i * 3] = 0
      particleColors[i * 3 + 1] = g
      particleColors[i * 3 + 2] = Math.random() * 0.3
    }

    const particleGeom = new THREE.BufferGeometry()
    particleGeom.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3))
    particleGeom.setAttribute('color', new THREE.BufferAttribute(particleColors, 3))

    const particleMat = new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })

    const particles = new THREE.Points(particleGeom, particleMat)
    scene.add(particles)

    // ── Ambient light ─────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x003311, 0.5))

    sceneRef.current = { scene, camera, renderer, helixGroup, particles, frameId: 0 }

    // ── Animation loop ────────────────────────────────────────
    const clock = new THREE.Clock()

    function animate() {
      const ref = sceneRef.current
      if (!ref) return
      ref.frameId = requestAnimationFrame(animate)

      const elapsed = clock.getElapsedTime()

      // Slow rotation of the helix
      ref.helixGroup.rotation.y = elapsed * 0.15
      ref.helixGroup.rotation.x = Math.sin(elapsed * 0.05) * 0.1

      // Particle drift
      ref.particles.rotation.y = elapsed * 0.02
      ref.particles.rotation.x = elapsed * 0.01

      ref.renderer.render(ref.scene, ref.camera)
    }

    animate()

    // ── Resize handler ────────────────────────────────────────
    function onResize() {
      const ref = sceneRef.current
      if (!ref) return
      const w = window.innerWidth
      const h = window.innerHeight
      ref.camera.aspect = w / h
      ref.camera.updateProjectionMatrix()
      ref.renderer.setSize(w, h)
    }

    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      const ref = sceneRef.current
      if (ref) {
        cancelAnimationFrame(ref.frameId)
        ref.renderer.dispose()
        mountRef.current?.removeChild(ref.renderer.domElement)
      }
    }
  }, [])

  // ── Scroll-driven camera movement ──────────────────────────
  useEffect(() => {
    const ref = sceneRef.current
    if (!ref) return

    // Camera zooms in and orbits as user scrolls
    const baseZ = 30
    const targetZ = 8
    ref.camera.position.z = baseZ - scrollProgress * (baseZ - targetZ)
    ref.camera.position.y = scrollProgress * -20
    ref.camera.position.x = Math.sin(scrollProgress * Math.PI) * 10

    // Helix opacity fades as we scroll past it
    ref.helixGroup.traverse((child) => {
      if (child instanceof THREE.Line || child instanceof THREE.Mesh) {
        const mat = child.material as THREE.Material
        mat.opacity = Math.max(0.1, 1 - scrollProgress * 1.5)
      }
    })
  }, [scrollProgress])

  return (
    <div
      ref={mountRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  )
}
