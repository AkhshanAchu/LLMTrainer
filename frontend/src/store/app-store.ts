import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AppState {
  selectedModelRepoId: string | null;
  selectedDatasetId: number | null;
  activeTrainingRunId: number | null;
  setSelectedModel: (repoId: string | null) => void;
  setSelectedDataset: (id: number | null) => void;
  setActiveTrainingRun: (id: number | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      selectedModelRepoId: null,
      selectedDatasetId: null,
      activeTrainingRunId: null,
      setSelectedModel: (repoId) => set({ selectedModelRepoId: repoId }),
      setSelectedDataset: (id) => set({ selectedDatasetId: id }),
      setActiveTrainingRun: (id) => set({ activeTrainingRunId: id }),
    }),
    { name: "finetune-studio-app-state" }
  )
);
