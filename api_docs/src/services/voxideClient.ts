import { VoxideClient } from '@voxide/react';

// Retrieve publishable key from window (injected by aimlite serve) or Vite env
const getPublicKey = (): string => {
  if (typeof window !== 'undefined') {
    const win = window as unknown as { VOXIDE_PUBLIC_KEY?: string; VOXIDE_API_KEY?: string };
    if (win.VOXIDE_PUBLIC_KEY) return win.VOXIDE_PUBLIC_KEY;
    if (win.VOXIDE_API_KEY) return win.VOXIDE_API_KEY;
  }
  return import.meta.env.VITE_VOXIDE_PUBLIC_KEY || import.meta.env.VITE_VOXIDE_API_KEY || '';
};

export const voxideClient = new VoxideClient({
  publicKey: getPublicKey(),
  ui: {
    accentColor: '#0ea5e9',
    theme: 'dark',
    title: 'AIMLite AI Helper',
    subtitle: 'Live voice & text agent for inference and health',
    position: 'bottom-right',
    launcherLabel: 'AI Helper',
    starters: [
      'Check server health',
      'Run model prediction',
      'Show me POST /predict endpoint',
      'How do I serve an AIMLite model?',
    ],
  },
});

export function setupVoxideCapabilities(onSelectSection?: (sectionId: string) => void) {
  voxideClient.register({
    predict: {
      description: 'Execute model prediction or inference with input features (numerical array or JSON).',
      params: {
        features: {
          type: 'object',
          required: false,
          description: 'Input payload or features to feed into the active model.',
        },
      },
      handler: async ({ features }) => {
        try {
          const body = features !== undefined ? JSON.stringify(features) : JSON.stringify({});
          const res = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body,
          });
          const data = await res.json();
          if (res.ok && data.status === 'success') {
            return {
              status: 'success',
              source: 'live_server',
              model: data.model,
              latency_ms: data.latency_ms,
              result: data.result,
            };
          } else {
            return {
              status: 'error',
              statusCode: res.status,
              error: data.message || 'Model inference returned an error.',
              details: data,
            };
          }
        } catch {
          // Server unreachable
        }

        if (onSelectSection) {
          onSelectSection('endpoint-predict');
        }

        return {
          status: 'offline',
          error: 'AIMLite inference server offline at http://127.0.0.1:8000. Start your server with "aimlite serve" to run live predictions.',
          endpoint: 'POST http://127.0.0.1:8000/predict',
        };
      },
    },

    health: {
      description: 'Check the health, status, and readiness of the AIMLite inference server.',
      params: {},
      handler: async () => {
        try {
          const res = await fetch('http://127.0.0.1:8000/health');
          if (res.ok) {
            const data = await res.json();
            if (onSelectSection) {
              onSelectSection('endpoint-health');
            }
            return {
              status: 'online',
              server: 'healthy',
              model: data.model || 'active_model',
              url: 'http://127.0.0.1:8000/health',
            };
          }
        } catch {
          // Server offline fallback
        }

        if (onSelectSection) {
          onSelectSection('endpoint-health');
        }

        return {
          status: 'offline',
          notice: 'AIMLite server is currently offline. Start it in terminal using "aimlite serve".',
          health_url: 'http://127.0.0.1:8000/health',
        };
      },
    },

    navigate: {
      description: 'Navigate to a documentation section (e.g. data, model, lifecycle, train, serve, rag, adapters, predict, health).',
      params: {
        destination: {
          type: 'string',
          required: true,
          description: 'The destination section or feature name to display.',
        },
      },
      handler: async ({ destination }) => {
        if (!onSelectSection) return { status: 'noop' };
        const term = String(destination).toLowerCase();

        let targetId = 'pillar-data';
        if (term.includes('predic')) targetId = 'endpoint-predict';
        else if (term.includes('health') || term.includes('status')) targetId = 'endpoint-health';
        else if (term.includes('serve') || term.includes('server')) targetId = 'cli-serve';
        else if (term.includes('train')) targetId = 'cli-train';
        else if (term.includes('eval')) targetId = 'cli-eval';
        else if (term.includes('init')) targetId = 'cli-init';
        else if (term.includes('doctor')) targetId = 'cli-doctor';
        else if (term.includes('rag') || term.includes('retriev')) targetId = 'paradigm-rag';
        else if (term.includes('adapter') || term.includes('lora')) targetId = 'paradigm-adapters';
        else if (term.includes('scratch')) targetId = 'paradigm-scratch';
        else if (term.includes('model')) targetId = 'pillar-model';
        else if (term.includes('life') || term.includes('cycle')) targetId = 'pillar-lifecycle';
        else if (term.includes('config')) targetId = 'foundations-config';

        onSelectSection(targetId);
        return { status: 'navigated', section: targetId };
      },
    },
  });
}
