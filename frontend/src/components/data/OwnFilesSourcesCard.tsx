/**
 * Data › Sources (step 1): "I already have train / validation / test files" (UI simplification pass,
 * phase 3). The same `PredefinedSplitCard` the Split step offers, wired with the values the split
 * form would send (`useOwnFilesWiring`), so a user with ready-made files imports them on the first
 * screen and goes straight to Train. The Split-step entry stays.
 */

import { useState } from 'react';

import type { PredefinedSplitManifest } from '../../api/predefinedSplit';
import PredefinedSplitCard from './PredefinedSplitCard';
import { CHAT_TEMPLATES, useOwnFilesWiring } from './ownFilesWiring';

interface Props {
    projectId: number;
    openSection?: string | null;
    onContinueToTrain: () => void | Promise<void>;
}

export default function OwnFilesSourcesCard({ projectId, openSection = null, onContinueToTrain }: Props) {
    const { wiring, loaded, reload } = useOwnFilesWiring(projectId);
    // A template the user picks here is sent, as on the Split step (the split form marks it touched).
    const [pickedTemplate, setPickedTemplate] = useState<string | null>(null);
    const [imported, setImported] = useState<PredefinedSplitManifest | null>(null);

    return (
        <section className="card" data-testid="own-files-sources">
            <h3 className="pdsplit__title">I already have train / validation / test files</h3>
            {!loaded ? (
                <p className="pdsplit__note" data-testid="own-files-sources-loading">Loading the project's split settings...</p>
            ) : (
                <PredefinedSplitCard
                    projectId={projectId}
                    chatTemplates={CHAT_TEMPLATES}
                    chatTemplate={pickedTemplate ?? wiring.chatTemplate}
                    sendChatTemplate={wiring.sendChatTemplate || pickedTemplate !== null}
                    onChatTemplateChange={setPickedTemplate}
                    ownTestRows={wiring.predefinedActive ? wiring.activeTestRows : null}
                    adapter={wiring.adapter}
                    openSection={openSection}
                    onImported={(manifest) => {
                        setImported(manifest);
                        reload();
                    }}
                />
            )}
            {imported && (
                <div className="pdsplit__continue" data-testid="own-files-continue">
                    <button type="button" className="btn btn-primary" data-testid="own-files-continue-train" onClick={() => void onContinueToTrain()}>
                        Continue to Train
                    </button>
                    <span className="pdsplit__note">
                        Your files are split already: Prepare, Labels, Generate and Split are optional.
                    </span>
                </div>
            )}
        </section>
    );
}
