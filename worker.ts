// src/worker.ts
//cloudflare Worker w/ SQLite-backed Durable Object for Crown Mural
//compat with Cloudflare Pages + Durable Objects
export interface Env {
  MURAL_STATE: DurableObjectNamespace;
}

/**
 * MuralStateObject - SQLite-backed Durable Object - rovides persistent state management for:
 * - user region selections
 * - collab viewing (multiple users see same state)
 * - analytics/tracking (optional)
 * 
 *atm OPTIONAL for current app alr working fully client-side.
 *enable if you want persistent or shared state features.
 */
export class MuralStateObject {
  private state: DurableObjectState;
  private sql: SqlStorage;
  private initialized = false;

  constructor(state: DurableObjectState) {
    this.state = state;
    this.sql = state.storage.sql;
  }

  /**
   *init SQLite schema if not exists
   */
  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    
    await this.state.blockConcurrencyWhile(async () => {
      //create tables for mural state
      this.sql.exec(`
        CREATE TABLE IF NOT EXISTS selections (
          user_id TEXT PRIMARY KEY,
          region_id INTEGER,
          selected_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS region_stats (
          region_id INTEGER PRIMARY KEY,
          view_count INTEGER DEFAULT 0,
          click_count INTEGER DEFAULT 0,
          last_viewed TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_selections_region 
          ON selections(region_id);
      `);
      
      this.initialized = true;
    });
  }

  async fetch(request: Request): Promise<Response> {
    await this.ensureInitialized();
    
    const url = new URL(request.url);
    const method = request.method;

    try {
      //get /state - get current shared state
      if (method === 'GET' && url.pathname === '/state') {
        return this.handleGetState();
      }

      //post /state - update user selection
      if (method === 'POST' && url.pathname === '/state') {
        return this.handleUpdateState(request);
      }

      //get /stats - get region stats
      if (method === 'GET' && url.pathname === '/stats') {
        return this.handleGetStats(url);
      }

      //post /track - Track a view or click event
      if (method === 'POST' && url.pathname === '/track') {
        return this.handleTrackEvent(request);
      }

      // DELETE /state - Clear user's selection
      if (method === 'DELETE' && url.pathname === '/state') {
        return this.handleClearState(request);
      }

      return new Response('Not found', { status: 404 });
    } catch (error) {
      console.error('MuralStateObject error:', error);
      return new Response(
        JSON.stringify({ error: 'Internal server error' }), 
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }
  }

  /**
   * Get all current selections (for collaborative viewing)
   */
  private handleGetState(): Response {
    const selections = this.sql.exec(`
      SELECT user_id, region_id, selected_at 
      FROM selections 
      ORDER BY selected_at DESC
    `).toArray();

    return new Response(
      JSON.stringify({ selections }), 
      { headers: { 'Content-Type': 'application/json' } }
    );
  }

  /**
   * Update a user's region selection
   */
  private async handleUpdateState(request: Request): Promise<Response> {
    const data = await request.json() as { 
      userId?: string; 
      regionId?: number | null;
    };

    if (!data.userId) {
      return new Response(
        JSON.stringify({ error: 'userId is required' }), 
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    if (data.regionId === null || data.regionId === undefined) {
      // Clear selection
      this.sql.exec(
        `DELETE FROM selections WHERE user_id = ?`,
        data.userId
      );
    } else {
      //upsert selection
      this.sql.exec(`
        INSERT INTO selections (user_id, region_id, selected_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET 
          region_id = excluded.region_id,
          selected_at = CURRENT_TIMESTAMP
      `, data.userId, data.regionId);
    }

    return new Response(
      JSON.stringify({ success: true }), 
      { headers: { 'Content-Type': 'application/json' } }
    );
  }

  /**
   * Get statistics for a specific region or all regions
   */
  private handleGetStats(url: URL): Response {
    const regionId = url.searchParams.get('regionId');

    let stats;
    if (regionId) {
      stats = this.sql.exec(
        `SELECT * FROM region_stats WHERE region_id = ?`,
        parseInt(regionId)
      ).toArray();
    } else {
      stats = this.sql.exec(
        `SELECT * FROM region_stats ORDER BY click_count DESC LIMIT 100`
      ).toArray();
    }

    return new Response(
      JSON.stringify({ stats }), 
      { headers: { 'Content-Type': 'application/json' } }
    );
  }

  /**
   * Track a view or click event for analytics
   */
  private async handleTrackEvent(request: Request): Promise<Response> {
    const data = await request.json() as {
      regionId?: number;
      eventType?: 'view' | 'click';
    };

    if (!data.regionId || !data.eventType) {
      return new Response(
        JSON.stringify({ error: 'regionId and eventType required' }), 
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    const column = data.eventType === 'click' ? 'click_count' : 'view_count';
    
    this.sql.exec(`
      INSERT INTO region_stats (region_id, ${column}, last_viewed)
      VALUES (?, 1, CURRENT_TIMESTAMP)
      ON CONFLICT(region_id) DO UPDATE SET
        ${column} = ${column} + 1,
        last_viewed = CURRENT_TIMESTAMP
    `, data.regionId);

    return new Response(
      JSON.stringify({ success: true }), 
      { headers: { 'Content-Type': 'application/json' } }
    );
  }

  /**
   * Clear a user's selection
   */
  private async handleClearState(request: Request): Promise<Response> {
    const data = await request.json() as { userId?: string };
    
    if (data.userId) {
      this.sql.exec(`DELETE FROM selections WHERE user_id = ?`, data.userId);
    }

    return new Response(
      JSON.stringify({ success: true }), 
      { headers: { 'Content-Type': 'application/json' } }
    );
  }
}

/**
 * Main worker fetch handler
 * Routes API requests to Durable Object, serves static assets via Pages
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    //handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
          'Access-Control-Max-Age': '86400',
        }
      });
    }

    //route API calls to Durable Object
    if (url.pathname.startsWith('/api/mural')) {
      //get / create singleton Durable Object
      const id = env.MURAL_STATE.idFromName('global-mural-state');
      const stub = env.MURAL_STATE.get(id);
      
      //forward request to DO, stripping /api/mural prefix
      const doUrl = new URL(request.url);
      doUrl.pathname = url.pathname.replace('/api/mural', '') || '/';
      
      const doRequest = new Request(doUrl.toString(), request);
      const response = await stub.fetch(doRequest);
      
      //add CORS headers to response
      const headers = new Headers(response.headers);
      headers.set('Access-Control-Allow-Origin', '*');
      
      return new Response(response.body, {
        status: response.status,
        headers
      });
    }

    //forCloudflare Pages, static assets are handled automatically
    //this worker only handles the /api/* routes
    //if deployed as pure Worker (not Pages), add asset handling here
    
    return new Response('Not found', { status: 404 });
  }
};
