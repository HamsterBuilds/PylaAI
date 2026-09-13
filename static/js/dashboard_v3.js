/* Reference-matched Hamster Bot views. Core data and API behavior stays in app.js. */
function renderNav() {
    const items = [
        ["dashboard", "dashboard", "Home"],
        ["queue", "queue", "Brawlers"],
        ["history", "history", "Analytics"],
        ["logs", "logs", "Output"],
        ["settings", "settings", "Settings"],
    ];
    const nav = document.querySelector(".nav-menu");
    if (!nav) return;
    nav.innerHTML = items.map(([view, icon, label]) => `
        <button class="nav-item ${view === state.currentView ? "active" : ""}" data-view="${view}" aria-current="${view === state.currentView ? "page" : "false"}" aria-label="${label}">
            <span class="nav-icon">${dashboardNavIcon(icon)}</span><span class="nav-label">${label}</span>
        </button>`).join("");
}

function analyticsIcon(kind) {
    const paths = {
        trophy: '<path d="M8 4h8v3c0 3-1.6 5.5-4 6.5C9.6 12.5 8 10 8 7V4Z"/><path d="M8 6H5v1c0 2 1.2 3 3.3 3.4M16 6h3v1c0 2-1.2 3-3.3 3.4M12 14v3M8 20h8M9 17h6"/>',
        win: '<path d="m6.5 12 3.4 3.4L18 7.5"/>',
        loss: '<path d="m7 7 10 10M17 7 7 17"/>',
        game: '<path d="M8 9h8a5 5 0 0 1 4.6 7l-.6 1.5a2 2 0 0 1-3.2.8L14.5 16h-5l-2.3 2.3a2 2 0 0 1-3.2-.8L3.4 16A5 5 0 0 1 8 9Z"/><path d="M8 12v4M6 14h4M16 13h.01M18 15h.01"/>',
        calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 10h18"/>'
    };
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[kind] || ''}</svg>`;
}

const dashboardUI = { metric: "trophies", allBrawlers: false, allMatches: false, logFilter: "all", logSearch: "", paused: false, verbose: false, logs: [], pendingLogs: [] };

function dashboardMatches() {
    return (state.bootstrap.history.items || []).flatMap(item => (item.matches || item.recent_matches || []).map(match => ({...match, brawler: item.brawler, icon_url: item.icon_url})))
        .sort((a,b) => String(b.date_sort || b.date_time).localeCompare(String(a.date_sort || a.date_time)));
}

function dashboardChart() {
    const points = (state.bootstrap.history.items || []).flatMap(item => (item.trophy_points || []).map(point => ({...point, brawler:item.brawler})))
        .sort((a,b) => String(a.label).localeCompare(String(b.label)));
    if (!points.length) return '<div class="v3-chart-empty">Your progress will appear here after your first recorded match.</div>';
    let running = 0;
    const values = [0, ...points.map(point => {
        running += dashboardUI.metric === "matches" ? (['victory','defeat'].includes(point.result) ? 1 : 0) : dashboardUI.metric === "wins" ? (point.result === "victory" ? 1 : point.result === "defeat" ? -1 : 0) : Number(point.delta || 0);
        return running;
    })];
    const low = Math.min(0,...values), high = Math.max(1,...values), span = high-low;
    const coords = values.map((v,i) => [i/(values.length-1)*1000, 180-(v-low)/span*170]);
    const line = coords.map(([x,y],i) => `${i?'L':'M'}${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
    return `<div class="v3-chart-body"><div class="v3-y-axis">${[1,.75,.5,.25,0].map(t=>`<span>${Math.round(low+span*t)}</span>`).join('')}</div><svg viewBox="0 0 1000 190" preserveAspectRatio="none" role="img" aria-label="${dashboardUI.metric === 'trophies' ? 'Cumulative trophy change' : dashboardUI.metric === 'wins' ? 'Cumulative wins minus losses' : 'Cumulative matches'}"><defs><linearGradient id="v3Fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3488ff" stop-opacity=".28"/><stop offset="1" stop-color="#3488ff" stop-opacity=".03"/></linearGradient></defs><path class="grid" d="M0 10H1000M0 52H1000M0 95H1000M0 137H1000M0 180H1000M0 0V180M200 0V180M400 0V180M600 0V180M800 0V180M1000 0V180"/><path class="fill" d="${line} L1000 190 L0 190Z"/><path class="line" d="${line}" vector-effect="non-scaling-stroke"/>${coords.slice(1).map(([x,y],i)=>`<circle class="v3-chart-point" cx="${x}" cy="${y}" r="5" tabindex="-1"><title>${escapeHtml(points[i].label)} · ${escapeHtml(points[i].brawler)}: ${values[i+1]}</title></circle>`).join('')}</svg></div><footer><span>${escapeHtml(points[0].label)}</span><span>${escapeHtml(points.at(-1).label)}</span></footer>`;
}

function renderHistory() {
    const view = document.getElementById("view-history");
    const summary = getHistorySummary();
    const items = [...(state.bootstrap.history.items || [])].sort((a,b)=>Number(b.total_matches||0)-Number(a.total_matches||0));
    const trophyGain = items.reduce((sum,item)=>sum+Number(item.trophy_delta||0),0);
    const top = dashboardUI.allBrawlers ? items : items.slice(0,5);
    const matches = dashboardMatches();
    const recent = dashboardUI.allMatches ? matches : matches.slice(0,5);
    const runtime = state.bootstrap.runtime || {};
    const duration = runtime.session_started_at ? formatSessionDuration(runtime.session_started_at) : "0m";
    const session = state.bootstrap.history.session_summary || {};
    const statRows = [["game","Matches Played",session.total_matches||0],["trophy","Trophies Gained",formatSignedNumber(session.trophy_delta||0)],["win","Wins",session.wins||0],["loss","Losses",session.losses||0],["win","Best Win Streak",Math.max(0,...items.map(i=>Number(i.best_win_streak||0)))],["trophy","Best Brawler",items.length?escapeHtml([...items].sort((a,b)=>b.trophy_delta-a.trophy_delta)[0].brawler):"—"]];
    const empty = '<p class="v3-empty">No matches recorded for this period.</p>';
    view.innerHTML = `
        <header class="v3-analytics-head"><div><h1>Analytics</h1><p>Track your progress and view detailed statistics.</p></div><label class="v3-date-filter">${analyticsIcon("calendar")}<select id="v3HistoryPeriod" aria-label="History period">${!dashboardUI.period && (state.historyStartDate || state.historyEndDate) ? `<option value="custom">${escapeHtml(historyDateRangeLabel())}</option>` : ""}<option value="all">All time</option><option value="today">Today</option><option value="week">Last 7 days</option><option value="month">Last 30 days</option></select></label></header>
        <div class="v3-analytics-kpis">${[["trophy","Trophies Gained",formatSignedNumber(trophyGain),"Net trophy change"],["win","Wins",summary.wins,`${formatPercent(summary.win_rate)} win rate`],["loss","Losses",summary.losses,formatPercent(summary.loss_rate)],["game","Matches Played",summary.total_matches,"Recorded matches"]].map(([icon,label,value,detail])=>`<article><i>${analyticsIcon(icon)}</i><div><p>${label}</p><strong>${value}</strong><em>${detail}</em></div></article>`).join('')}</div>
        <section class="v3-trophy-chart"><div><h2>${dashboardUI.metric==='trophies'?'Trophies Over Time':dashboardUI.metric==='wins'?'Wins / Losses Over Time':'Matches Over Time'}</h2><span class="v3-chart-tabs">${[["trophies","Trophies"],["wins","Wins / Losses"],["matches","Matches"]].map(([key,label])=>`<button data-metric="${key}" class="${dashboardUI.metric===key?'active':''}" aria-pressed="${dashboardUI.metric===key}">${label}</button>`).join('')}</span></div>${dashboardChart()}</section>
        <section class="v3-most"><div><h2>Most Played Brawlers</h2><button id="v3AllBrawlers">${dashboardUI.allBrawlers?'Show Less':'View All'}</button></div><div class="v3-list">${top.length?top.map((item,index)=>`<button class="v3-most-row" data-history-brawler="${escapeHtml(item.brawler)}"><b>${index+1}</b><img src="${escapeHtml(item.icon_url)}" alt=""><span><strong>${escapeHtml(item.brawler)}</strong><small>${item.total_matches||0} matches</small></span><em>${formatSignedNumber(item.trophy_delta||0)} 🏆</em></button>`).join(''):empty}</div></section>
        <section class="v3-session"><h2>Session Stats <span>◷ ${duration}</span></h2><dl>${statRows.map(([icon,label,value])=>`<div><dt><i class="stat-icon ${icon}">${analyticsIcon(icon)}</i>${label}</dt><dd>${value}</dd></div>`).join('')}</dl></section>
        <section class="v3-recent"><h2>Recent Matches <button id="v3AllMatches">${dashboardUI.allMatches?'Show Less':'View All'}</button></h2><div class="v3-list">${recent.length?recent.map(item=>`<article><img src="${escapeHtml(item.icon_url)}" alt="${escapeHtml(item.brawler)}"><p><strong class="${item.result==='victory'?'green':item.result==='defeat'?'red':'muted'}">${escapeHtml(item.result==='victory'?'Victory':item.result==='defeat'?'Defeat':item.result)}</strong><span>${escapeHtml(item.playstyle_gamemodes?.join(', ') || 'Trophy Match')}</span></p><b class="${Number(item.trophy_delta)>=0?'green':'red'}">${formatSignedNumber(item.trophy_delta)} 🏆</b><time>${escapeHtml(item.date_time)}</time></article>`).join(''):empty}</div></section>
        <section class="v3-playtime"><h2>Playtime <span title="Elapsed time since the current session started">ⓘ</span></h2><strong>${duration}</strong><em>${runtime.is_running?'Current session':'Session elapsed time'}</em><div class="v3-playtime-note">${runtime.session_started_at?'Started '+escapeHtml(new Date(runtime.session_started_at*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})):'Start a session to track your playtime.'}<p>Daily playtime history is not recorded yet.</p></div></section>`;
    const period = document.getElementById('v3HistoryPeriod');
    period.value = dashboardUI.period || (state.historyStartDate || state.historyEndDate ? 'custom' : 'all');
    period.addEventListener('change', async () => {
        dashboardUI.period = period.value;
        const end = new Date(), start = new Date();
        start.setDate(start.getDate()-(period.value==='week'?6:period.value==='month'?29:0));
        const iso = date => `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
        await applyHistoryDateFilter(period.value==='all'?'':iso(start), period.value==='all'?'':iso(end));
        renderHistory();
    });
    view.querySelectorAll('[data-metric]').forEach(button=>button.addEventListener('click',()=>{dashboardUI.metric=button.dataset.metric;renderHistory();}));
    document.getElementById('v3AllBrawlers').onclick=()=>{dashboardUI.allBrawlers=!dashboardUI.allBrawlers;renderHistory();};
    document.getElementById('v3AllMatches').onclick=()=>{dashboardUI.allMatches=!dashboardUI.allMatches;renderHistory();};
    view.querySelectorAll('[data-history-brawler]').forEach(button=>button.onclick=()=>openHistoryDetails(button.dataset.historyBrawler));
}

function updateHistoryViewData() { renderHistory(); }

function logLevel(line) {
    const explicit = line.match(/\[(INFO|SUCCESS|WARNING|WARN|ERROR|DEBUG|TRACE|STDERR)\]/i);
    if (explicit) {
        const level = explicit[1].toLowerCase();
        return {warn: 'warning', trace: 'debug', stderr: 'error'}[level] || level;
    }
    return /error|defeat|stderr|exception|failed/i.test(line) ? "error" : /success|victory|connected|loaded/i.test(line) ? "success" : /warn/i.test(line) ? "warning" : /debug|trace/i.test(line) ? "debug" : "info";
}

function outputSide() {
    const runtime = state.bootstrap.runtime || {}, queue = state.bootstrap.queue || [];
    const active = queue.find(item=>item.brawler === runtime.current_brawler) || queue[0];
    const session = state.bootstrap.history.session_summary || {};
    const status = ['paused','pausing','stopping'].includes(runtime.state) ? runtime.state : runtime.is_running ? 'running' : 'idle';
    const target = Number(active?.push_until || 0), current = Number(active?.current_value ?? active?.trophies ?? 0);
    const progress = target ? Math.min(100, Math.max(0, Math.round(current/target*100))) : 0;
    const completed = queue.filter(item=>Number(item.current_value ?? item.trophies ?? 0)>=Number(item.push_until||1000)).length;
    return `<section class="v3-status-card"><div class="v3-card-title"><h2>Current Status</h2><b class="${status==='running'?'running':''}">● ${status}</b></div>
        ${active?`<div class="v3-current-brawler"><img src="${escapeHtml(active.icon_url)}" alt=""><div><h3>${escapeHtml(active.brawler)}</h3><strong>🏆 ${current} / ${target}</strong><div class="v3-current-progress"><div class="v3-progress"><i style="width:${progress}%"></i></div><span>${progress}%</span></div></div></div>`:'<p class="v3-empty">Add a brawler to your queue to get started.</p>'}
        <dl>${[['♟','Current Task',escapeHtml(runtime.current_stage || (runtime.is_running?'Bot active':'Waiting to start'))],['🎮','Match Type',active?.target_type==='wins'?'Target Wins':'Trophy Match'],['◷','Session Time',runtime.session_started_at?formatSessionDuration(runtime.session_started_at):'0m'],['⟳','Matches Played',session.total_matches||0],['🏆','Net Trophies',formatSignedNumber(session.trophy_delta||0)],['✓','Wins / Losses',`${session.wins||0} / ${session.losses||0}`]].map(([icon,label,value])=>`<div><dt><i>${icon}</i>${label}</dt><dd>${value}</dd></div>`).join('')}</dl></section>
        <section class="v3-queue-progress"><div class="v3-card-title"><h2>Queue Progress</h2><button id="v3ViewQueue">View Queue</button></div><div class="v3-queue-row">${queue.slice(0,5).map(item=>`<button data-output-brawler="${escapeHtml(item.brawler)}" class="${item===active?'active':''}"><img src="${escapeHtml(item.icon_url)}" alt=""><strong>${escapeHtml(item.brawler)}</strong><span>${Number(item.current_value??item.trophies??0)} / ${Number(item.push_until||1000)}</span></button>`).join('')||'<p class="v3-empty">Your queue is empty.</p>'}</div><div class="v3-queue-footer"><div class="v3-progress"><i style="width:${queue.length?completed/queue.length*100:0}%"></i></div><span>${completed} / ${queue.length} completed</span></div></section>`;
}

function updateOutputLines() {
    const terminal = document.getElementById('logsTerminal');
    if (!terminal) return;
    const scroller = terminal.parentElement;
    const atBottom = scroller.scrollHeight-scroller.clientHeight-scroller.scrollTop < 50;
    const visible = dashboardUI.logs.filter(line => {
        const level=logLevel(line);
        return (dashboardUI.verbose || level!=='debug') && (dashboardUI.logFilter==='all'||dashboardUI.logFilter===level) && line.toLowerCase().includes(dashboardUI.logSearch.toLowerCase());
    });
    const markup = visible.length ? visible.map(line=>`<div class="log-line v3-log-line ${logLevel(line)}">${escapeHtml(line)}</div>`).join('') : `<div class="v3-empty-log">${dashboardUI.logs.length?'No logs match your filters.':'No output yet. Start Hamster Bot to see live actions.'}</div>`;
    if (terminal.innerHTML !== markup) terminal.innerHTML=markup;
    if (!dashboardUI.paused && (atBottom || state.forceScrollLogs)) scroller.scrollTop=scroller.scrollHeight;
    state.forceScrollLogs=false;
}

function renderLogsContent(logs) {
    const view=document.getElementById('view-logs');
    if (!view) return;
    dashboardUI.pendingLogs=logs.map(String);
    if (!dashboardUI.paused) dashboardUI.logs=dashboardUI.pendingLogs;
    if (!view.querySelector('.v3-page-head')) {
        view.innerHTML=`<header class="v3-page-head"><div><h1>Output</h1><p>Live logs, bot actions and detailed output.</p></div><div class="v3-log-filters" aria-label="Log severity">${['All','Info','Success','Warning','Error'].map(label=>`<button data-log-filter="${label.toLowerCase()}">${label}</button>`).join('')}</div><label class="v3-log-search">${iconMarkup('search')}<input id="v3LogSearch" type="search" aria-label="Search logs" placeholder="Search logs..."></label><button id="btnScrollToggle" class="v3-pause-output"></button></header>
        <div class="v3-output-grid"><section class="v3-terminal-card"><div class="logs-terminal-container" tabindex="0" aria-label="Live output"><div id="logsTerminal" class="logs-terminal"></div></div><footer><select id="v3LogDetail" aria-label="Log detail"><option value="normal">Normal</option><option value="verbose">Verbose</option></select><span></span><button id="btnCopyLogs">${iconMarkup('copy')} Copy Logs</button><button id="btnSaveLogs">${iconMarkup('import')} Export</button><button id="btnClearLogs">${iconMarkup('trash')} Clear</button></footer></section><aside class="v3-output-side"></aside></div>`;
        document.getElementById('v3LogSearch').value=dashboardUI.logSearch;
        document.getElementById('v3LogSearch').oninput=event=>{dashboardUI.logSearch=event.target.value;updateOutputLines();};
        document.getElementById('v3LogDetail').value=dashboardUI.verbose?'verbose':'normal';
        document.getElementById('v3LogDetail').onchange=event=>{dashboardUI.verbose=event.target.value==='verbose';updateOutputLines();};
        view.querySelectorAll('[data-log-filter]').forEach(button=>button.onclick=()=>{dashboardUI.logFilter=button.dataset.logFilter;syncOutputControls();updateOutputLines();});
        document.getElementById('btnScrollToggle').onclick=()=>{dashboardUI.paused=!dashboardUI.paused;if(!dashboardUI.paused){dashboardUI.logs=dashboardUI.pendingLogs;state.forceScrollLogs=true;}syncOutputControls();updateOutputLines();};
        document.getElementById('btnCopyLogs').onclick=copyAllLogs;
        document.getElementById('btnSaveLogs').onclick=saveLogsToTxt;
        document.getElementById('btnClearLogs').onclick=async()=>{try{const result=await fetchJSON('/api/runtime/logs',{method:'DELETE'});if(result.ok){dashboardUI.logs=[];dashboardUI.pendingLogs=[];updateOutputLines();showToast('Logs cleared.','success');}}catch(error){showToast('Failed to clear logs.','error');}};
    }
    syncOutputControls();
    updateOutputLines();
    const side=view.querySelector('.v3-output-side');
    side.innerHTML=outputSide();
    document.getElementById('v3ViewQueue').onclick=()=>setView('queue');
    side.querySelectorAll('[data-output-brawler]').forEach(button=>button.onclick=()=>{state.selectedBrawler=button.dataset.outputBrawler;setView('queue');});
}

function syncOutputControls() {
    document.querySelectorAll('[data-log-filter]').forEach(button=>{const selected=button.dataset.logFilter===dashboardUI.logFilter;button.classList.toggle('active',selected);button.setAttribute('aria-pressed',String(selected));});
    const pause=document.getElementById('btnScrollToggle');
    if(pause){pause.innerHTML=`${iconMarkup(dashboardUI.paused?'play':'pause')} ${dashboardUI.paused?'Resume Output':'Pause Output'}`;pause.setAttribute('aria-pressed',String(dashboardUI.paused));}
}

function dashboardNavIcon(kind) {
    const paths = {
        dashboard: '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z"/>',
        queue: '<path fill-rule="evenodd" d="M12 2a10 10 0 0 0-7 17.1V21h4v-3h6v3h4v-1.9A10 10 0 0 0 12 2ZM7.5 8a2.5 3 0 1 0 0 6 2.5 3 0 0 0 0-6Zm9 0a2.5 3 0 1 0 0 6 2.5 3 0 0 0 0-6ZM12 13l-2 3h4Z"/>',
        history: '<rect x="2" y="12" width="5" height="10" rx="1.5"/><rect x="9.5" y="3" width="5" height="19" rx="1.5"/><rect x="17" y="8" width="5" height="14" rx="1.5"/>',
        logs: '<path d="m4 5 7 7-7 7" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><rect x="13" y="18" width="9" height="3" rx="1"/>',
        settings: '<path fill-rule="evenodd" d="m10 2-1 3-2 1-3-.5-2 4 2.2 2v2L2 15.5l2 4 3-.5 2 1 1 3h4l1-3 2-1 3 .5 2-4-2.2-2v-2l2.2-2-2-4-3 .5-2-1-1-3Zm2 6a4 4 0 1 1 0 8 4 4 0 0 1 0-8Z"/>'
    };
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[kind] || ''}</svg>`;
}
