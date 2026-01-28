// worker-configuration.d.ts
//type definitions for Cloudflare Worker environment

interface Env {
  MURAL_STATE: DurableObjectNamespace;
}

//extend global types for Durable Objects with SQLite
interface DurableObjectState {
  storage: DurableObjectStorage & {
    sql: SqlStorage;
  };
  blockConcurrencyWhile<T>(callback: () => Promise<T>): Promise<T>;
}

interface SqlStorage {
  exec(query: string, ...params: unknown[]): SqlStorageResult;
}

interface SqlStorageResult {
  toArray(): Record<string, unknown>[];
}
