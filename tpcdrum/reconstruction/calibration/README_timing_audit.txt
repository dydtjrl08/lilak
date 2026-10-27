TPCDrum timing-input collection

This is a generic public-code guide. Actual ROOT metadata exports, server paths,
SSH addresses, logs, UUIDs and data-content hashes belong in local collections;
none are embedded here. Collection is not physical drift calibration.

1. Use a fresh source clone rather than pulling into a working simulation.
Set these environment variables to your own existing paths:
  RECO_CODE: fresh clone's tpcdrum/reconstruction directory
  RECO_REPO: original initialized simulation checkout
  RECO_DIGI: actual digitized ROOT input
  RECO_ASSET_ROOT: an optional candidate response/transport folder
  RECO_OUTPUT: a NEW output directory

2. Run in the environment where ROOT/LILAK works:

  python3 "$RECO_CODE/calibration/export_collect_inputs.py" \
    --repo "$RECO_REPO" \
    --parameters "$RECO_REPO/tpcdrum/simulation/SimParameters.mac" \
    --digi "$RECO_DIGI" \
    --asset-root "$RECO_ASSET_ROOT" \
    --output "$RECO_OUTPUT"

The collector itself uses Python's standard library; no pandas, Torch or rg.
Omit --asset-root if no candidate folder is known; it may be repeated.
--stored-parameter-json accepts separately exported native/Uproot metadata and
may also be repeated. The native metadata export from --digi is read automatically.
Original files are read-only. The full digi/waveforms are never copied.

assets/ contains current-config references.
assets_stored_parameters/ contains stored-ROOT references separately.
assets_unverified_candidates/ contains explicit same-name candidates separately.
A candidate does not erase a missing historical reference or establish byte identity.
Missing, unreadable, oversized or conflicting inputs remain in the manifest/logs.
The default per-asset cap can be changed with --max-asset-mb in a NEW collection.
Existing output directories/archives are refused.

3. Optional actual initialized material/geometry export:
Append --geant4-material-export to the same command. It reuses the original
simulation's build settings without replacing the executable, changing gas/GEM
settings or generating events. The target server build still needs validation.
Keep any compile/link/export failure logs with the collection.

4. Transfer the resulting archive using your own SSH endpoint, for example:

  scp -P "$RECO_SSH_PORT" \
    "$RECO_SSH_TARGET:$RECO_REMOTE_ARCHIVE" "$RECO_LOCAL_NEW_FOLDER/"

Create a new local destination and preserve missing/error information.
No server address, username or local private directory is supplied by this guide.

5. Interpretation:
Stored settings establish what the ROOT file records, not successful asset loading
or the response of a historical binary. Pulse-template mean shift, finite drift
steps, field distortion, time walk, clipping and trigger reference must be checked
using the same time statistic/preprocessing as inference. A common clock offset
is correlated across tracks and remains relevant for absolute vertex position.
Same-name files are candidates until their response/provenance is validated.
No drift conversion or detector-precision claim is enabled by collection alone.
