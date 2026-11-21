/**
 * Training mode (Train › More options › Advanced): SFT is the default; continued pretraining on
 * documents, DPO and ORPO are chosen here. Reads/writes the shared config form; TrainingPanel owns
 * the payload (the field is still `training_mode`, sent exactly as before).
 */

import type { TrainingConfigForm } from './useTrainingConfigForm';

export default function TrainingModeSection({ form }: { form: TrainingConfigForm }) {
  const { trainingMode, setTrainingMode, setTaskType, setTouchedConfig } = form;
  return (
    <div className="training-grid-2">
      <div className="form-group">
        <label className="form-label" htmlFor="training-mode-select">Training Mode</label>
        <select
          id="training-mode-select"
          className="input"
          value={trainingMode}
          onChange={(e) => {
            const nextMode = e.target.value;
            setTrainingMode(nextMode);
            setTouchedConfig((prev) => ({ ...prev, training_mode: true }));
            if (nextMode === 'dpo' || nextMode === 'orpo') {
              setTaskType('causal_lm');
              setTouchedConfig((prev) => ({ ...prev, task_type: true }));
            }
          }}
        >
          <option value="sft">SFT</option>
          <option value="domain_pretrain">Continued pretraining (documents)</option>
          <option value="dpo">DPO</option>
          <option value="orpo">ORPO</option>
        </select>
      </div>
    </div>
  );
}
