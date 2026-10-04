// Data for the patient view: calls to the local MedGemma helpers, and fixed,
// reviewable wording for groups and where to look for help.  Anything a
// patient reads about which doctor or service to approach comes from these
// tables, not from the language model.

import { api } from './source'

export interface InterpretedTerm {
  /** The clinical term MedGemma read from the patient's words. */
  term: string
  /** HPO terms it may correspond to, best first; empty when none was found. */
  matches: { id: string; label: string }[]
}

export async function patientStatus(): Promise<'ready' | 'offline'> {
  try {
    const body = await api<{ medgemma: 'ready' | 'offline' }>('patient/status')
    return body.medgemma
  } catch {
    return 'offline'
  }
}

const post = <T>(path: string, body: unknown) =>
  api<T>(`patient/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const interpretSymptoms = (text: string) => post<{ terms: InterpretedTerm[] }>('interpret', { text })
export const explainDisease = (id: string, name: string) => post<{ text: string | null }>('explain-disease', { id, name })
export const explainGroup = (category: string, examples: string[], shared: string[]) =>
  post<{ text: string }>('explain-group', { category, examples, shared })

/** Plain-language name, a one-line description, and who usually looks after it, per Orphanet category. */
export interface GroupInfo {
  name: string
  about: string
  doctors: string
}

const GROUPS: Record<string, GroupInfo> = {
  'Rare bone disease': {
    name: 'Bone and skeleton conditions',
    about: 'Conditions that affect how bones grow, their shape or their strength.',
    doctors: 'orthopaedics, rheumatology or a clinical genetics (skeletal dysplasia) service',
  },
  'Rare cardiac disease': {
    name: 'Heart conditions',
    about: 'Conditions that affect the heart muscle, valves or rhythm.',
    doctors: 'cardiology, or an inherited cardiac conditions clinic',
  },
  'Rare circulatory system disease': {
    name: 'Blood vessel conditions',
    about: 'Conditions that affect arteries, veins or lymph vessels.',
    doctors: 'vascular medicine or cardiology',
  },
  'Rare developmental defect during embryogenesis': {
    name: 'Conditions present from birth that affect development',
    about: 'Conditions that start before birth and can affect growth, learning or how the body forms.',
    doctors: 'clinical genetics and paediatrics',
  },
  'Rare disorder due to toxic effects': {
    name: 'Conditions caused by harmful substances',
    about: 'Conditions caused by contact with medicines, chemicals or other substances.',
    doctors: 'your GP, who can refer you to toxicology or the right specialist',
  },
  'Rare endocrine disease': {
    name: 'Hormone conditions',
    about: 'Conditions that affect the glands that make hormones.',
    doctors: 'endocrinology',
  },
  'Rare gastroenterologic disease': {
    name: 'Digestive system conditions',
    about: 'Conditions that affect the stomach, bowel or digestion.',
    doctors: 'gastroenterology',
  },
  'Rare gynecologic or obstetric disease': {
    name: 'Gynaecological and pregnancy-related conditions',
    about: 'Conditions that affect the womb, ovaries or pregnancy.',
    doctors: 'gynaecology or obstetrics',
  },
  'Rare hematologic disease': {
    name: 'Blood conditions',
    about: 'Conditions that affect blood cells, clotting or bone marrow.',
    doctors: 'haematology',
  },
  'Rare hepatic disease': {
    name: 'Liver conditions',
    about: 'Conditions that affect the liver or bile ducts.',
    doctors: 'hepatology',
  },
  'Rare immune disease': {
    name: 'Immune system conditions',
    about: 'Conditions where the immune system is too weak or attacks the body.',
    doctors: 'immunology',
  },
  'Rare inborn error of metabolism': {
    name: 'Inherited metabolic conditions',
    about: 'Inherited conditions where the body cannot process certain substances properly.',
    doctors: 'a metabolic medicine service or clinical genetics',
  },
  'Rare infectious disease': {
    name: 'Rare infections',
    about: 'Uncommon infections caused by bacteria, viruses, fungi or parasites.',
    doctors: 'infectious diseases',
  },
  'Rare infertility': {
    name: 'Fertility conditions',
    about: 'Conditions that affect the ability to have children.',
    doctors: 'reproductive medicine or a fertility clinic',
  },
  'Rare neoplastic disease': {
    name: 'Rare tumours and cancers',
    about: 'Uncommon growths, some cancerous and some not.',
    doctors: 'oncology, often at a specialist cancer centre',
  },
  'Rare neurologic disease': {
    name: 'Brain, nerve and muscle conditions',
    about: 'Conditions that affect the brain, spinal cord, nerves or muscles.',
    doctors: 'neurology (paediatric neurology for children)',
  },
  'Rare odontologic disease': {
    name: 'Teeth and mouth conditions',
    about: 'Conditions that affect the teeth, gums or mouth.',
    doctors: 'a dentist or oral medicine specialist',
  },
  'Rare ophthalmic disorder': {
    name: 'Eye conditions',
    about: 'Conditions that affect the eyes or sight.',
    doctors: 'ophthalmology (an eye specialist)',
  },
  'Rare otorhinolaryngologic disease': {
    name: 'Ear, nose and throat conditions',
    about: 'Conditions that affect hearing, the nose, throat or voice.',
    doctors: 'ear, nose and throat (ENT) or audiology',
  },
  'Rare renal disease': {
    name: 'Kidney conditions',
    about: 'Conditions that affect the kidneys.',
    doctors: 'nephrology (a kidney specialist)',
  },
  'Rare respiratory disease': {
    name: 'Lung and breathing conditions',
    about: 'Conditions that affect the lungs or breathing.',
    doctors: 'respiratory medicine',
  },
  'Rare skin disease': {
    name: 'Skin conditions',
    about: 'Conditions that affect the skin, hair or nails.',
    doctors: 'dermatology',
  },
  'Rare systemic or rheumatologic disease': {
    name: 'Conditions affecting several parts of the body',
    about: 'Conditions that affect joints, connective tissue or several organs at once.',
    doctors: 'rheumatology',
  },
  'Rare urogenital disease': {
    name: 'Urinary and genital conditions',
    about: 'Conditions that affect the bladder, urinary tract or genitals.',
    doctors: 'urology',
  },
  'Rare allergic disease': {
    name: 'Allergic conditions',
    about: 'Conditions where the body reacts strongly to things that are usually harmless.',
    doctors: 'allergy or immunology',
  },
}

const SURGICAL: GroupInfo = {
  name: 'Conditions usually treated with surgery',
  about: 'Conditions affecting body structures that are often treated by a surgeon.',
  doctors: 'your GP, who can refer you to the right surgical team',
}
const OTHER: GroupInfo = {
  name: 'Other rare conditions',
  about: 'Rare conditions that do not fit one body system.',
  doctors: 'clinical genetics, through a referral from your GP',
}

export function groupInfo(category: string): GroupInfo {
  if (GROUPS[category]) return GROUPS[category]
  return /surgical/i.test(category) ? SURGICAL : OTHER
}

/** General places to find reliable information and support. */
export const RESOURCES: { name: string; url: string; about: string }[] = [
  {
    name: 'Orphanet',
    url: 'https://www.orpha.net',
    about: 'The rare disease reference: information on each condition, expert centres and patient organisations.',
  },
  {
    name: 'Genetic Alliance UK',
    url: 'https://geneticalliance.org.uk',
    about: 'UK charity supporting people with genetic and rare conditions.',
  },
  {
    name: 'EURORDIS – Rare Diseases Europe',
    url: 'https://www.eurordis.org',
    about: 'A network of patient organisations across Europe.',
  },
  {
    name: 'NORD',
    url: 'https://rarediseases.org',
    about: 'US National Organization for Rare Disorders: condition reports and patient support.',
  },
]
