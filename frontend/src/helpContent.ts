// ─────────────────────────────────────────────────────────────────────────────
// MARINA – help tooltip content
//
// Every help blurb shown in the UI lives here. Edit the strings below to
// update what users see when they hover a  ?  button. Newlines (\n) are
// rendered as line breaks in the tooltip.
// ─────────────────────────────────────────────────────────────────────────────

export const HELP = {

  // ── Header controls ─────────────────────────────────────────────────────────

  controls: {
    model:
      'The neural network model used to embed spectral data and SMILES strings ' +
      'into a shared fingerprint space. Different models may have been trained on ' +
      'different datasets or architectures. The default model is selected automatically ' +
      'and works well for most queries.',

    status:
      'Shows whether the backend server is reachable and whether a model is fully ' +
      'loaded in memory.\n\n' +
      '• Ready — a model is loaded and predictions can be submitted immediately.\n' +
      '• Loading — the server is reachable but still loading the model.\n' +
      '• Offline — the backend is not responding; check that the containers are running.',

    usage:
      'Cumulative usage since counting began: total queries served (predictions, ' +
      'SMILES searches and custom cards) and the number of distinct visitors.\n\n' +
      'Visitors are counted by a salted, one-way hash of the IP address — no ' +
      'addresses are stored, and the hashes cannot be reversed into a visitor list.',
  },

  // ── Mode tabs ────────────────────────────────────────────────────────────────

  tabs: {
    spectral:
      'Use this mode when you have raw spectral data (NMR peaks or mass spec ' +
      'intensities) for an unknown compound. Enter your peaks into the spreadsheet ' +
      'and MARINA will embed them with a neural network, then retrieve the most ' +
      'structurally similar compounds from the database.',

    smiles:
      'Use this mode when you already know a candidate chemical structure. Enter ' +
      'a SMILES string and MARINA will embed it with the same neural network used ' +
      'for spectral queries, then retrieve structurally similar database compounds. ' +
      'Useful for structure–structure similarity searches.',
  },

  // ── Spectral panel ───────────────────────────────────────────────────────────

  spectral: {
    spreadsheet:
      'The spreadsheet accepts four types of spectral data:\n\n' +
      '• HSQC — three columns per row: ¹H shift (ppm), ¹³C shift (ppm), intensity. ' +
      'All three values are required per row.\n' +
      '• ¹H NMR — one column: chemical shift (ppm).\n' +
      '• ¹³C NMR — one column: chemical shift (ppm).\n' +
      '• Mass Spec — two columns: m/z and intensity. Both values required per row.\n\n' +
      'Leave any section empty if you don\'t have that data type. ' +
      'You can paste data directly from Excel or a spreadsheet app.',

    preview:
      'Renders whatever is currently in the spreadsheet, updating as you type or paste. ' +
      'Only complete rows are plotted, so partially-filled rows will not appear.\n\n' +
      '• HSQC — cross-peak map with ¹H on the x-axis and ¹³C on the y-axis, both ' +
      'running high→low ppm (origin bottom-right). Peaks are colored by the sign of ' +
      'the intensity, the phase convention of a multiplicity-edited HSQC: positive is ' +
      'CH/CH₃, negative is CH₂.\n' +
      '• ¹H / ¹³C NMR — peak positions on a high→low ppm axis. These columns carry no ' +
      'intensity, so all sticks are drawn at the same height.\n' +
      '• MS/MS — m/z increasing left→right, intensities normalized to the base peak.\n\n' +
      'Hover any plot to read off the nearest peak.',

    mw:
      'The molecular weight of the compound in Daltons (exact or nominal). ' +
      'This value is passed to the neural network as an additional input feature ' +
      'to improve the fingerprint — it does not filter results. Leave blank if unknown.',

    mwFilter:
      'Restricts retrieved results to compounds whose recorded mass falls within ' +
      'this range. Both bounds are optional — leave either blank for no bound on ' +
      'that side. This filter is applied after neural network retrieval and does ' +
      'not affect the fingerprint itself.\n\n' +
      'Compounds with no mass recorded in the database are kept rather than ' +
      'filtered out, so a sparsely-annotated entry cannot be hidden by the filter.',

    resultsCount:
      'How many top-ranked compounds to retrieve from the database (1–50). ' +
      'Results are ranked by cosine similarity in fingerprint space. ' +
      'Higher values show more candidates but may take slightly longer.',

    predict:
      'Encodes your spectral data through the MARINA neural network to produce a ' +
      'molecular fingerprint, then retrieves the k most similar compounds by cosine ' +
      'similarity. At least one spreadsheet column must contain valid values.',

    examples:
      'Loads a bundled example dataset into the spreadsheet so you can explore ' +
      'MARINA\'s output without your own data. The examples are real natural product ' +
      'structures with associated spectral data.',
  },

  // ── SMILES panel ─────────────────────────────────────────────────────────────

  smiles: {
    input:
      'Enter a valid SMILES string for your query structure ' +
      '(e.g. CC(C)CCO for 3-methyl-1-butanol). ' +
      'The string is canonicalized on the server before embedding — ' +
      'you do not need to normalize it manually. Press Enter to search.',

    search:
      'Encodes the SMILES structure using the MARINA neural network and retrieves ' +
      'the k most similar database compounds by cosine similarity in fingerprint space. ' +
      'Note: the similarity score reflects embedding space distance, not Tanimoto similarity.',
  },

  // ── Custom SMILES panel ──────────────────────────────────────────────────────

  custom: {
    panel:
      'Score any SMILES structure against the fingerprint produced by the most ' +
      'recent Spectral Data prediction or SMILES Search. This lets you check how ' +
      'similar a specific candidate is to your query without running a new search. ' +
      'The result appears as a highlighted card alongside the main results.',
  },

} as const
