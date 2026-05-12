/**
 * ScaffoldForm — generate a contract-valid plugin scaffold from the
 * UI (priority.md P38 + P40). Posts to ``/api/extensions/scaffold``
 * with the active kind + the form fields. The parent receives the
 * response and renders the file preview / download buttons.
 */

import { useCallback, useEffect, useState } from 'react';

import type {
    PluginKind,
    ScaffoldRequest,
    ScaffoldResponse,
} from '../../types/extensions';
import { PLUGIN_KIND_LABEL } from '../../types/extensions';

interface Props {
    kind: PluginKind;
    onGenerated: (response: ScaffoldResponse) => void;
    onError: (message: string) => void;
    submit: (body: ScaffoldRequest) => Promise<ScaffoldResponse>;
}

interface FormState {
    plugin_id: string;
    display_name: string;
    description: string;
    author: string;
    version: string;
    write: boolean;
}

const EMPTY_STATE: FormState = {
    plugin_id: '',
    display_name: '',
    description: '',
    author: '',
    version: '',
    write: true,
};

export default function ScaffoldForm({
    kind,
    onGenerated,
    onError,
    submit,
}: Props) {
    const [form, setForm] = useState<FormState>(EMPTY_STATE);
    const [submitting, setSubmitting] = useState(false);

    // Reset the form when the operator switches plugin kind.
    useEffect(() => {
        setForm(EMPTY_STATE);
    }, [kind]);

    const update = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
        setForm((prev) => ({ ...prev, [key]: value }));
    }, []);

    const handleSubmit = useCallback(
        async (event: React.FormEvent<HTMLFormElement>) => {
            event.preventDefault();
            const plugin_id = form.plugin_id.trim();
            if (!plugin_id) {
                onError('Plugin id is required.');
                return;
            }

            const body: ScaffoldRequest = { kind, plugin_id, write: form.write };
            if (form.display_name.trim()) body.display_name = form.display_name.trim();
            if (form.description.trim()) body.description = form.description.trim();
            if (form.author.trim()) body.author = form.author.trim();
            if (form.version.trim()) body.version = form.version.trim();

            setSubmitting(true);
            try {
                const response = await submit(body);
                onGenerated(response);
            } catch (err) {
                const message =
                    err instanceof Error
                        ? err.message
                        : 'Scaffold generation failed.';
                onError(message);
            } finally {
                setSubmitting(false);
            }
        },
        [form, kind, onError, onGenerated, submit],
    );

    return (
        <form className="scaffold-form" onSubmit={handleSubmit}>
            <div className="scaffold-form-row">
                <label className="scaffold-form-label" htmlFor="scaffold-plugin-id">
                    Plugin id
                </label>
                <input
                    id="scaffold-plugin-id"
                    className="input"
                    type="text"
                    placeholder={`e.g. my-${kind.replace('_', '-')}`}
                    value={form.plugin_id}
                    onChange={(event) => update('plugin_id', event.target.value)}
                    required
                    autoComplete="off"
                />
            </div>
            <div className="scaffold-form-row">
                <label className="scaffold-form-label" htmlFor="scaffold-display-name">
                    Display name
                </label>
                <input
                    id="scaffold-display-name"
                    className="input"
                    type="text"
                    placeholder={`${PLUGIN_KIND_LABEL[kind]} plugin`}
                    value={form.display_name}
                    onChange={(event) => update('display_name', event.target.value)}
                />
            </div>
            <div className="scaffold-form-row">
                <label className="scaffold-form-label" htmlFor="scaffold-description">
                    Description
                </label>
                <textarea
                    id="scaffold-description"
                    className="input"
                    rows={2}
                    placeholder="One-line summary that ends up in the docstring + README."
                    value={form.description}
                    onChange={(event) => update('description', event.target.value)}
                />
            </div>
            <div className="scaffold-form-row scaffold-form-row-split">
                <div>
                    <label className="scaffold-form-label" htmlFor="scaffold-author">
                        Author
                    </label>
                    <input
                        id="scaffold-author"
                        className="input"
                        type="text"
                        placeholder="optional"
                        value={form.author}
                        onChange={(event) => update('author', event.target.value)}
                    />
                </div>
                <div>
                    <label className="scaffold-form-label" htmlFor="scaffold-version">
                        Version
                    </label>
                    <input
                        id="scaffold-version"
                        className="input"
                        type="text"
                        placeholder="0.1.0"
                        value={form.version}
                        onChange={(event) => update('version', event.target.value)}
                    />
                </div>
            </div>
            <div className="scaffold-form-row scaffold-form-row-inline">
                <label className="scaffold-form-label" htmlFor="scaffold-write">
                    <input
                        id="scaffold-write"
                        type="checkbox"
                        checked={form.write}
                        onChange={(event) => update('write', event.target.checked)}
                    />{' '}
                    Write files to DATA_DIR/extension_scaffolds
                </label>
            </div>
            <div className="scaffold-form-actions">
                <button
                    type="submit"
                    className="btn btn-primary btn-sm"
                    disabled={submitting}
                >
                    {submitting ? 'Generating…' : 'Generate scaffold'}
                </button>
            </div>
        </form>
    );
}
