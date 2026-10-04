// The single boundary between the UI and wherever data comes from.  Components
// only talk to a DataSource, so wiring in a real backend means adding an
// implementation here, not touching the views.

import { decodeCatalogue, type FeatureCatalogue } from './features'
import type { LiteratureGraph } from './types'

export interface DataSource {
  /** Literature-backed relationship graph. */
  loadLiteratureGraph(): Promise<LiteratureGraph>
  /**
   * Feature profiles for catalogue diseases (the columns evaluation.py
   * compares), or null when they have not been generated yet.
   */
  loadFeatureCatalogue(): Promise<FeatureCatalogue | null>
}

const dataUrl = (file: string) => `${import.meta.env.BASE_URL}data/${file}`

/** Reads the static JSON produced by the scripts in frontend/scripts/. */
export class StaticDataSource implements DataSource {
  async loadLiteratureGraph(): Promise<LiteratureGraph> {
    const response = await fetch(dataUrl('literature_graph.json'))
    if (!response.ok) {
      throw new Error(
        `Could not load literature graph (${response.status}). ` +
          'Run `python3 frontend/scripts/build_literature_graph.py` from the repository root.',
      )
    }
    return response.json()
  }

  async loadFeatureCatalogue(): Promise<FeatureCatalogue | null> {
    const response = await fetch(dataUrl('disease_features.json'))
    // The dev server answers missing files with index.html, so check the type.
    if (!response.ok || !response.headers.get('content-type')?.includes('json')) return null
    return decodeCatalogue(await response.json())
  }
}

export const dataSource: DataSource = new StaticDataSource()
