/**
 * Optional "Output schema (JSON)" for the json-extraction task type.
 *
 * Declares the fields of the JSON record the model must produce — their
 * types and, for category-like fields, the allowed values — so evaluation
 * can score null vs. wrong-value, flag out-of-vocabulary categories and
 * type violations. Saved on the project's selected recipe. Leaving it
 * empty keeps the field list derived from the answer key.
 */

import { useState } from 'react';
import { saveRecipeOutputSchema } from '../../api/recipes';
import './OutputSchemaEditor.css';

interface OutputSchemaEditorProps {
    projectId: number;
    /** The schema currently saved on the project (null/undefined = none). */
    initialSchema?: Record<string, unknown> | null;
    onSaved?: (schema: Record<string, unknown> | null) => void;
}

export const OUTPUT_SCHEMA_EXAMPLE = `{
  "required": ["damage", "injuries_fatal"],
  "properties": {
    "damage": {"type": ["string", "null"], "enum": ["destroyed", "substantial", "minor", "none", null]},
    "injuries_fatal": {"type": ["integer", "null"]}
  }
}`;

function errorText(err: unknown): string {
    const body = (err as { response?: { data?: { detail?: unknown; message?: unknown } } })?.response?.data;
    if (typeof body?.detail === 'string' && body.detail) return body.detail;
    if (typeof body?.message === 'string' && body.message) return body.message;
    return err instanceof Error ? err.message : 'Could not save the output schema.';
}

function pretty(schema: Record<string, unknown> | null | undefined): string {
    return schema && Object.keys(schema).length ? JSON.stringify(schema, null, 2) : '';
}

export default function OutputSchemaEditor({ projectId, initialSchema, onSaved }: OutputSchemaEditorProps) {
    const [text, setText] = useState(pretty(initialSchema));
    const [saved, setSaved] = useState(pretty(initialSchema));
    const [error, setError] = useState('');
    const [saving, setSaving] = useState(false);

    const dirty = text.trim() !== saved.trim();

    const save = async () => {
        setError('');
        const trimmed = text.trim();
        let payload: Record<string, unknown> | null = null;
        if (trimmed) {
            try {
                const parsed: unknown = JSON.parse(trimmed);
                if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
                    setError('The output schema must be a JSON object with a "properties" map.');
                    return;
                }
                payload = parsed as Record<string, unknown>;
            } catch (err) {
                setError(`Not valid JSON: ${err instanceof Error ? err.message : 'parse error'}`);
                return;
            }
        }
        setSaving(true);
        try {
            await saveRecipeOutputSchema(projectId, payload);
            const next = pretty(payload);
            setText(next);
            setSaved(next);
            onSaved?.(payload);
        } catch (err) {
            setError(errorText(err));
        } finally {
            setSaving(false);
        }
    };

    return (
        <section className="card output-schema-editor" data-testid="output-schema-editor">
            <h3 className="output-schema-editor__title">Output schema (JSON), optional</h3>
            <p className="u-hint u-m0">
                Name each field of the record and give its type; add <code>enum</code> for category
                fields. Evaluation then reports null precision and recall, out-of-vocabulary values and
                type violations per field. Leave it empty to use the fields found in your answer key.
            </p>
            <label className="form-label" htmlFor="output-schema-text">Schema</label>
            <textarea
                id="output-schema-text"
                className="input output-schema-editor__text"
                rows={10}
                spellCheck={false}
                value={text}
                placeholder={OUTPUT_SCHEMA_EXAMPLE}
                onChange={(e) => setText(e.target.value)}
                data-testid="output-schema-text"
            />
            <p className="u-meta u-m0">
                Types: string, integer, number, boolean, null, object, array (types are checked strictly:
                the text <code>"3"</code> is not an integer). Use <code>["string", "null"]</code> when the text may not state the field.
            </p>
            <p className="u-meta u-m0" data-testid="output-schema-apply-hint">
                Changes apply the next time the dataset is prepared; retrain afterwards so the training
                prompt lists the same fields.
            </p>
            {error && (
                <div className="error-banner" role="alert" data-testid="output-schema-error">{error}</div>
            )}
            <div className="output-schema-editor__actions">
                <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    disabled={saving || !dirty}
                    onClick={() => void save()}
                    data-testid="output-schema-save"
                >
                    {saving ? 'Saving…' : text.trim() ? 'Save output schema' : 'Clear output schema'}
                </button>
                {!dirty && saved && <span className="u-meta" data-testid="output-schema-saved">Saved</span>}
            </div>
        </section>
    );
}
