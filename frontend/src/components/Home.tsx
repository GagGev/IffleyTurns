import type { AppView } from '../lib/route'
import { ThemeToggle } from './ThemeToggle'

/** Landing page: choose the patient or the researcher view. */
export function Home({ onChoose }: { onChoose: (view: AppView) => void }) {
  return (
    <div className="home">
      <div className="home-top">
        <ThemeToggle />
      </div>
      <header className="home-header">
        <h1>Rare Disease Explorer</h1>
        <p>
          Rare diseases are individually uncommon but together affect millions of people. This tool maps how thousands of
          rare diseases resemble each other, using their symptoms, genes, treatments and medical descriptions.
        </p>
      </header>
      <div className="home-choices">
        <button type="button" className="home-card" onClick={() => onChoose('patient')}>
          <span className="home-card-title">Patient view</span>
          <span className="home-card-body">
            Describe symptoms in your own words and see which groups of rare conditions share them, explained simply, with
            ideas for where to look for help.
          </span>
          <span className="home-card-cta">Start →</span>
        </button>
        <button type="button" className="home-card" onClick={() => onChoose('research')}>
          <span className="home-card-title">Researcher view</span>
          <span className="home-card-body">
            Explore the full similarity graph of 7,500 rare diseases
          </span>
          <span className="home-card-cta">Open the explorer →</span>
        </button>
      </div>
      <p className="home-note">
        This is a research tool. It cannot diagnose any condition. Always talk to a doctor about health concerns.
      </p>
    </div>
  )
}
