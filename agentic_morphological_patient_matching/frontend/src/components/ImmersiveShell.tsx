import React, { useState, useEffect, useRef, useCallback } from 'react'
import DNAHelix from './DNAHelix'

interface Section {
  id: string
  label: string
  icon: string
}

const SECTIONS: Section[] = [
  { id: 'hero', label: 'Clinical PAL', icon: '🧬' },
  { id: 'data', label: 'Data', icon: '📊' },
  { id: 'explore', label: 'Explore', icon: '🔍' },
  { id: 'cluster', label: 'Morphology Groups', icon: '🧩' },
  { id: 'retrieve', label: 'Patient Matcher', icon: '🔗' },
  { id: 'chat', label: 'Chat', icon: '💬' },
  { id: 'biomarkers', label: 'Biomarker Discovery', icon: '🧪' },
]

interface ImmersiveShellProps {
  children: React.ReactNode[]
  activeSection: string
  onSectionChange: (id: string) => void
}

export default function ImmersiveShell({
  children,
  activeSection,
  onSectionChange,
}: ImmersiveShellProps) {
  const [scrollProgress, setScrollProgress] = useState(0)
  const [heroVisible, setHeroVisible] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return
    const el = containerRef.current
    const maxScroll = el.scrollHeight - el.clientHeight
    const progress = maxScroll > 0 ? el.scrollTop / maxScroll : 0
    setScrollProgress(Math.min(1, progress))
    setHeroVisible(el.scrollTop < window.innerHeight * 0.5)

    // Determine active section based on scroll position
    const sectionEls = el.querySelectorAll('[data-section]')
    for (let i = sectionEls.length - 1; i >= 0; i--) {
      const rect = sectionEls[i].getBoundingClientRect()
      if (rect.top <= window.innerHeight * 0.4) {
        const id = sectionEls[i].getAttribute('data-section')
        if (id) onSectionChange(id)
        break
      }
    }
  }, [onSectionChange])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('scroll', handleScroll, { passive: true })
    return () => el.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  const scrollToSection = (id: string) => {
    const el = containerRef.current?.querySelector(`[data-section="${id}"]`)
    el?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <>
      {/* 3D Background */}
      <DNAHelix scrollProgress={scrollProgress} />

      {/* Matrix rain overlay */}
      <div className="matrix-overlay" />

      {/* Floating nav dots */}
      <nav className="immersive-nav">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            className={`nav-dot ${activeSection === s.id ? 'active' : ''}`}
            onClick={() => scrollToSection(s.id)}
            title={s.label}
          >
            <span className="nav-dot-icon">{s.icon}</span>
            <span className="nav-dot-label">{s.label}</span>
          </button>
        ))}
      </nav>

      {/* Scrollable content */}
      <div ref={containerRef} className="immersive-scroll-container">
        {/* Hero section */}
        <section data-section="hero" className="immersive-hero">
          <div
            className="hero-content"
            style={{ opacity: heroVisible ? 1 : 0, transform: `translateY(${scrollProgress * -100}px)` }}
          >
            <div className="hero-badge">AWS Life Sciences Symposium</div>
            <h1 className="hero-title">
              <span className="hero-glow">Clinical</span> PAL
            </h1>
            <p className="hero-subtitle">
              Precision AI for Life Sciences
            </p>
            <p className="hero-description">
              Biomarker discovery and clinical trial patient screening
              <br />
              powered by BioFMs, AI agents, and real-world oncology data
            </p>
            <button className="hero-cta" onClick={() => scrollToSection('data')}>
              <span>Enter the Lab</span>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M10 3v14m0 0l-5-5m5 5l5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        </section>

        {/* Panel sections */}
        {children.map((child, i) => {
          const section = SECTIONS[i + 1] // skip hero
          if (!section) return null
          return (
            <section
              key={section.id}
              data-section={section.id}
              className="immersive-section"
            >
              <div className="section-header">
                <span className="section-icon">{section.icon}</span>
                <h2 className="section-title">{section.label}</h2>
                <div className="section-line" />
              </div>
              <div className="section-content">{child}</div>
            </section>
          )
        })}

        {/* Footer */}
        <footer className="immersive-footer">
          <div className="footer-helix">🧬</div>
          <p>Clinical PAL — Built with Amazon Bedrock, Kiro, and the HANCOCK Dataset</p>
        </footer>
      </div>
    </>
  )
}
