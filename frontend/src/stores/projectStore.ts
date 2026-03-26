import { create } from 'zustand';
import api from '../api/client';
import type { Project, PipelineStatusResponse, TabKey } from '../types';

interface ProjectState {
    // Project list
    projects: Project[];
    totalProjects: number;
    isLoadingProjects: boolean;

    // Active project
    activeProject: Project | null;
    pipelineStatus: PipelineStatusResponse | null;
    activeTab: TabKey;

    // Actions
    fetchProjects: () => Promise<void>;
    createProject: (
        name: string,
        description?: string,
        baseModel?: string,
        starterPackId?: string | null,
        domainPackId?: number | null,
        domainProfileId?: number | null,
    ) => Promise<Project>;
    fetchProject: (id: number) => Promise<void>;
    deleteProject: (id: number) => Promise<void>;
    fetchPipelineStatus: (projectId: number) => Promise<void>;
    setActiveTab: (tab: TabKey) => void;
    setActiveProject: (project: Project | null) => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
    projects: [],
    totalProjects: 0,
    isLoadingProjects: false,
    activeProject: null,
    pipelineStatus: null,
    activeTab: 'data',

    fetchProjects: async () => {
        set({ isLoadingProjects: true });
        try {
            const res = await api.get('/projects');
            set({
                projects: res.data.projects,
                totalProjects: res.data.total,
                isLoadingProjects: false,
            });
        } catch {
            set({ isLoadingProjects: false });
        }
    },

    createProject: async (
        name,
        description = '',
        baseModel = '',
        starterPackId = null,
        domainPackId = null,
        domainProfileId = null,
    ) => {
        const res = await api.post('/projects', {
            name,
            description,
            base_model_name: baseModel,
            starter_pack_id: starterPackId,
            domain_pack_id: domainPackId,
            domain_profile_id: domainProfileId,
        });
        const project = res.data;
        set((state) => ({
            projects: [project, ...state.projects],
            totalProjects: state.totalProjects + 1,
        }));
        return project;
    },

    fetchProject: async (id) => {
        const res = await api.get(`/projects/${id}`);
        set({ activeProject: res.data });
    },

    deleteProject: async (id) => {
        await api.delete(`/projects/${id}`);
        set((state) => ({
            projects: state.projects.filter((p) => p.id !== id),
            totalProjects: state.totalProjects - 1,
        }));
    },

    fetchPipelineStatus: async (projectId) => {
        const res = await api.get(`/projects/${projectId}/pipeline/status`);
        set({ pipelineStatus: res.data });
    },

    setActiveTab: (tab) => set({ activeTab: tab }),
    setActiveProject: (project) => set({ activeProject: project }),
}));
