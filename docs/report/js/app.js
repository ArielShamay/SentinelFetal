
// =========================================================================
// RENDERERS
// =========================================================================

function renderGapAnalysis() {
    const root = document.getElementById('gap-root');
    if (!root) return;

    let html = '';
    GAP_DATA.forEach(item => {
        let borderColor = 'var(--neon-green)';
        let pillClass = 'status-done';
        let pillText = 'תקין (Done)';

        if (item.status === 'partial') {
            borderColor = 'var(--neon-orange)';
            pillClass = 'status-partial';
            pillText = 'חלקי (Partial)';
        } else if (item.status === 'missing') {
            borderColor = 'var(--neon-red)';
            pillClass = 'status-missing';
            pillText = 'חסר (Missing)';
        }

        html += `
            <div class="gap-section" style="border-right: 5px solid ${borderColor};"> 
                <div class="gap-header">
                     <div class="status-pill ${pillClass}">${pillText}</div>
                     <div class="gap-title">${item.title}</div>
                </div>
                <div class="gap-body">
                    <p style="color:#ccc; font-size:1.1rem; line-height:1.6;">${item.desc}</p>
                </div>
            </div>
        `;
    });
    root.innerHTML = html;
}

function renderMegaFlow() {
    const root = document.getElementById('mega-root');
    if (!root) return;

    let html = '';
    PIPELINE.forEach((node, idx) => {
        html += `
            <div class="mega-node">
                <div class="mn-header">
                     <i class="fas ${node.icon} mn-icon" style="color:${node.color}"></i>
                     <div class="mn-title">${node.title}</div>
                </div>
                <div class="mn-body">
                    
                    <div class="viz-field">
                        <div class="vf-type" style="color:${node.color}">INPUT</div>
                        <div class="vf-name">${node.inputs.length} Fields</div>
                    </div>

                    <div style="flex:1; display:flex; align-items:center; justify-content:center; text-align:center; padding:10px;">
                        <p style="color:#eee; line-height:1.5;">${node.process}</p>
                    </div>

                    <div class="viz-field">
                         <div class="vf-type" style="color:${node.color}">OUTPUT</div>
                         <div class="vf-name">${node.outputs.length} Fields</div>
                    </div>
                </div>
            </div>
        `;

        if (idx < PIPELINE.length - 1) {
            html += `<div class="pipe-connector"></div>`;
        }
    });
    root.innerHTML = html;
}

// =========================================================================
// DOCS & STAGES
// =========================================================================

let currentFontSize = 18; // Increased default font size

function adjustFontSize(delta) {
    currentFontSize += delta;
    if (currentFontSize < 14) currentFontSize = 14;
    if (currentFontSize > 28) currentFontSize = 28;

    document.querySelectorAll('.markdown-body').forEach(el => {
        el.style.fontSize = currentFontSize + 'px';
    });
    document.querySelectorAll('.goal-text').forEach(el => {
        el.style.fontSize = (currentFontSize + 2) + 'px';
    });
}

function scrollToDoc(id) {
    const el = document.getElementById(id);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Update Active State for docs-nav-item
        document.querySelectorAll('.docs-nav-item').forEach(i => i.classList.remove('active'));
        const navItem = document.getElementById('nav-' + id);
        if (navItem) navItem.classList.add('active');

        // Update Active State for stage-nav-item
        document.querySelectorAll('.stage-nav-item').forEach(i => i.classList.remove('active'));
        const stageNavItem = document.getElementById('nav-' + id);
        if (stageNavItem) stageNavItem.classList.add('active');
    }
}

function renderDocs() {
    const root = document.getElementById('docs-root');
    const sidebar = document.getElementById('docs-sidebar-root');
    if (!root || !sidebar) return;

    if (typeof DOCS_STRUCTURE === 'undefined') {
        root.innerHTML = '<div style="color:red; padding:20px;">Error: docs_content.js not loaded.</div>';
        return;
    }

    let mainHtml = '';
    let sidebarHtml = '';

    // 1. Dashboard (README)
    if (typeof DOCS_INDEX !== 'undefined') {
        const id = 'doc-readme';
        sidebarHtml += `
            <div id="nav-${id}" class="docs-nav-item active" onclick="scrollToDoc('${id}')">
                <i class="fas fa-home"></i> דאשבורד סנטינל
            </div>
        `;

        mainHtml += `
            <div id="${id}" class="card-section" style="border:1px solid #444; background:#0f0f15;">
                <div class="markdown-body" style="padding:10px; font-size:${currentFontSize}px;">
                    ${marked.parse(DOCS_INDEX)}
                </div>
            </div>
        `;
    }

    // 2. Categories
    for (const [category, files] of Object.entries(DOCS_STRUCTURE)) {
        if (files.length === 0) continue;

        sidebarHtml += `<div class="docs-category-title">${category}</div>`;

        files.forEach((file, idx) => {
            const id = `doc-${category}-${idx}`;

            sidebarHtml += `
                <div id="nav-${id}" class="docs-nav-item" onclick="scrollToDoc('${id}')">
                    ${file.title}
                </div>
            `;

            mainHtml += `
                 <div id="${id}" class="card-section">
                    <div class="cs-title">
                        <i class="fas fa-file-alt" style="color:#aaa"></i>
                        ${file.title}
                    </div>
                    
                    <div class="markdown-body" style="color:#ccc; font-size:${currentFontSize}px;">
                        ${marked.parse(file.content)}
                    </div>
                 </div>
             `;
        });
    }

    root.innerHTML = mainHtml;
    sidebar.innerHTML = sidebarHtml;
}

// =========================================================================
// STAGES BREAKDOWN - PREMIUM VISUAL UPGRADE
// =========================================================================

function parseMarkdownToCards(markdown) {
    const sections = markdown.split(/^## /m);
    let html = '';

    sections.forEach((section, idx) => {
        if (!section.trim()) return;

        // --- 1. OVERVIEW & I/O SECTION ---
        if (idx === 0 && !markdown.startsWith('## ')) {
            // Extract Goal
            const goalMatch = section.match(/### מטרה\n([\s\S]*?)(?=###|$)/);
            const goalText = goalMatch ? goalMatch[1].trim() : '';

            // Extract Input
            const inputMatch = section.match(/### קלט\n([\s\S]*?)(?=###|$)/);
            const inputText = inputMatch ? inputMatch[1].trim() : '';

            // Extract Output
            const outputMatch = section.match(/### פלט[^\n]*\n([\s\S]*?)(?=---|$)/);
            const outputText = outputMatch ? outputMatch[1].trim() : '';

            if (goalText) {
                html += `
                    <div class="stage-card overview-hero">
                        <div class="sc-header"><i class="fas fa-bullseye"></i><span>GOAL & OBJECTIVE</span></div>
                        <div class="goal-text">${goalText}</div>
                    </div>
                `;
            }

            if (inputText || outputText) {
                html += `
                    <div class="io-grid">
                        <div class="io-card input">
                            <div class="sc-header"><i class="fas fa-sign-in-alt"></i><span>INPUTS</span></div>
                            <div class="markdown-body">${marked.parse(inputText)}</div>
                        </div>
                        <div class="io-center">
                            <div class="io-arrow"><i class="fas fa-arrow-right"></i></div>
                            <div class="io-label">PROCESS</div>
                        </div>
                        <div class="io-card output">
                             <div class="sc-header"><i class="fas fa-sign-out-alt"></i><span>OUTPUTS</span></div>
                             <div class="markdown-body">${marked.parse(outputText)}</div>
                        </div>
                    </div>
                `;
            } else if (!goalText) {
                // Fallback
                html += `
                    <div class="stage-card info">
                        <div class="markdown-body">${marked.parse(section)}</div>
                    </div>
                `;
            }

        } else {
            // --- 2. STANDARD SECTIONS (Build, Verify) ---
            const firstLineEnd = section.indexOf('\n');
            const headerText = section.substring(0, firstLineEnd).trim();
            section = '## ' + section;

            let cardType = 'info';
            let icon = 'fa-info-circle';

            if (headerText.toLowerCase().includes('build') || headerText.includes('בנייה') || headerText.toLowerCase().includes('pipeline')) {
                cardType = 'build';
                icon = 'fa-tools';
            } else if (headerText.toLowerCase().includes('verify') || headerText.toLowerCase().includes('certify') || headerText.includes('אימות')) {
                cardType = 'verify';
                icon = 'fa-shield-alt';
            }

            html += `
                <div class="stage-card ${cardType}">
                    <div class="sc-header">
                        <i class="fas ${icon}"></i>
                        <span>${cardType.toUpperCase()}</span>
                    </div>
                    <div class="markdown-body">
                        ${marked.parse(section)}
                    </div>
                </div>
            `;
        }
    });

    return html;
}

function renderStages() {
    const root = document.getElementById('stages-root');
    const sidebar = document.getElementById('stages-sidebar-root');
    if (!root || !sidebar) return;

    if (typeof STAGES_STRUCTURE === 'undefined') {
        root.innerHTML = '<div style="color:red; padding:20px;">Error: STAGES_STRUCTURE not loaded.</div>';
        return;
    }

    let mainHtml = '';
    let sidebarHtml = '';

    STAGES_STRUCTURE.forEach((file, idx) => {
        const id = `stage-${idx}`;
        const titleFull = file.title.replace('.md', '').replace(/_/g, ' ');
        // Extract Stage Number (SentinelFetal_Stage_2_Windowing -> 2)
        const match = titleFull.match(/Stage (\d+)/);
        const stageNum = match ? match[1] : (idx === 0 ? '0' : 'A');
        const stageName = titleFull
            .replace('SentinelFetal Stage', '')
            .replace(/\d+/, '')
            .replace(/^\s+/, '')
            .trim();

        sidebarHtml += `
            <div id="nav-${id}" class="stage-nav-item ${idx === 0 ? 'active' : ''}" onclick="scrollToDoc('${id}')">
                <div class="sni-num">${stageNum}</div>
                <div class="sni-content">
                    <div class="sni-label">STAGE ${stageNum}</div>
                    <div class="sni-title">${stageName}</div>
                </div>
            </div>
        `;

        mainHtml += `
             <div id="${id}" class="stage-container">
                 <div class="stage-banner">
                    <div class="sb-label">STAGE ${stageNum} ARCHITECTURE</div>
                    <div class="sb-title">${stageName}</div>
                 </div>
                 
                 ${parseMarkdownToCards(file.content)}
             </div>
             <div class="stage-separator"></div>
         `;
    });

    root.innerHTML = mainHtml;
    sidebar.innerHTML = sidebarHtml;
}


// =========================================================================
// ARCHITECTURE MAP - CODE FLOW VISUALIZATION (CYBERPUNK DASHBOARD)
// =========================================================================

let activeArchStage = 1; // Global state for active stage

function renderArchitectureMap() {
    const root = document.getElementById('arch-root');
    if (!root) return;

    if (typeof ARCHITECTURE_DATA === 'undefined') {
        root.innerHTML = '<div style="color:red; padding:20px;">Error: ARCHITECTURE_DATA not loaded.</div>';
        return;
    }

    let html = '';

    // === STAGE TABS (Navigation) ===
    html += `<div class="arch-stage-tabs" style="margin-top: 40px;">`;
    ARCHITECTURE_DATA.stages.forEach((stage) => {
        const isActive = stage.id === activeArchStage ? 'active' : '';
        html += `
            <button class="arch-tab-btn ${isActive}" 
                    onclick="switchArchTab(${stage.id})" 
                    data-stage="${stage.id}"
                    style="--tab-color: ${stage.color};">
                <div class="tab-icon"><i class="fas ${stage.icon}"></i></div>
                <div class="tab-content">
                    <div class="tab-label">STAGE ${stage.id}</div>
                    <div class="tab-title">${stage.title}</div>
                </div>
            </button>
        `;
    });
    html += `</div>`;

    // === DYNAMIC CONTENT AREA (Stage Details) ===
    html += `<div id="arch-stage-content" class="arch-stage-content"></div>`;

    root.innerHTML = html;

    // Render the initial stage
    renderStageDetail(activeArchStage);
}

function switchArchTab(stageId) {
    activeArchStage = stageId;

    // Update tab buttons
    document.querySelectorAll('.arch-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (parseInt(btn.dataset.stage) === stageId) {
            btn.classList.add('active');
        }
    });

    // Render the selected stage
    renderStageDetail(stageId);
}

function renderStageDetail(stageId) {
    const contentArea = document.getElementById('arch-stage-content');
    if (!contentArea) return;

    const stage = ARCHITECTURE_DATA.stages.find(s => s.id === stageId);
    if (!stage) return;

    let html = '';

    // Stage Header
    html += `
        <div class="stage-detail-header" style="border-color: ${stage.color};">
            <div class="sdh-icon" style="background: ${stage.color};">
                <i class="fas ${stage.icon}"></i>
            </div>
            <div class="sdh-info">
                <div class="sdh-label">STAGE ${stage.id}</div>
                <div class="sdh-title">${stage.title}</div>
                <div class="sdh-desc">${stage.description}</div>
                <div class="sdh-script">
                    <i class="fas fa-file-code"></i>
                    <code>${stage.script}</code>
                </div>
            </div>
        </div>
    `;

    // Sub-Steps Grid (3 columns max)
    html += `
        <div class="stage-section-title">
            <i class="fas fa-list-ol"></i>
            <span>Implementation Steps (${stage.steps.length})</span>
        </div>
        <div class="tech-cards-grid">
    `;

    stage.steps.forEach((step, idx) => {
        html += `
            <div class="tech-card" style="--card-color: ${stage.color};">
                <div class="tech-card-header">
                    <div class="tech-card-num">${idx + 1}</div>
                    <div class="tech-card-title">${step.name}</div>
                </div>
                <div class="tech-card-body">
                    <div class="tech-card-file">
                        <i class="fas fa-file-code"></i> ${step.file}
                    </div>
                    <div class="tech-card-functions">
                        ${step.functions.map(f => `<code>${f}</code>`).join('')}
                    </div>
                    <div class="tech-card-io">
                        <div class="io-row in">
                            <span class="io-icon">→</span>
                            <span class="io-text">${step.input}</span>
                        </div>
                        <div class="io-row out">
                            <span class="io-icon">←</span>
                            <span class="io-text">${step.output}</span>
                        </div>
                    </div>
                    <div class="tech-card-details">${step.details}</div>
                </div>
            </div>
        `;
    });

    html += `</div>`; // Close tech-cards-grid

    // Output Artifacts
    html += `
        <div class="stage-section-title">
            <i class="fas fa-download"></i>
            <span>Output Artifacts</span>
        </div>
        <div class="artifacts-grid">
    `;

    stage.outputs.forEach(out => {
        html += `
            <div class="artifact-pill" style="--pill-color: ${stage.color};">
                <code>${out.name}</code>
                <span>${out.desc}</span>
            </div>
        `;
    });

    html += `</div>`; // Close artifacts-grid

    // Fade-in animation
    contentArea.style.opacity = '0';
    contentArea.innerHTML = html;
    setTimeout(() => { contentArea.style.opacity = '1'; }, 50);
}

// =========================================================================
// PLAN NAVIGATION (Stage 1 Deep Dive)
// =========================================================================

function renderPlanNav() {
    const navRoot = document.getElementById('plan-nav-root');
    const contentRoot = document.getElementById('plan-content-root');
    if (!navRoot || !contentRoot) return;

    let navHtml = '';
    let contentHtml = '';

    PLAN_STEPS.forEach((step, idx) => {
        const isActive = idx === 0 ? 'active' : '';
        const display = idx === 0 ? 'block' : 'none';

        // Nav Item
        navHtml += `
            <div class="step-item ${isActive}" onclick="switchStep(${step.id})" id="nav-step-${step.id}">
                <div class="step-num">SUB-STEP 1.${step.id}</div>
                <div class="step-title">${step.title}</div>
            </div>
        `;

        // Content
        contentHtml += `
            <div id="content-step-${step.id}" class="step-content-box" style="display:${display};">
                <div class="detail-header">
                     <div class="dh-title">${step.title}</div>
                     <div class="dh-desc">${step.subtitle}</div>
                </div>
                ${step.content_he}
            </div>
        `;
    });

    navRoot.innerHTML = navHtml;
    contentRoot.innerHTML = contentHtml;
}

// =========================================================================
// LOGIC
// =========================================================================

function switchStep(id) {
    // Nav
    document.querySelectorAll('.step-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`nav-step-${id}`).classList.add('active');

    // Content
    document.querySelectorAll('.step-content-box').forEach(el => el.style.display = 'none');
    document.getElementById(`content-step-${id}`).style.display = 'block';
}

function setView(viewId) {
    document.querySelectorAll('.view-layer').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    document.getElementById('view-' + viewId).classList.add('active');

    const tabs = ['report', 'plan1', 'mega', 'stages', 'arch', 'docs'];
    const idx = tabs.indexOf(viewId);
    if (idx >= 0) document.querySelectorAll('.nav-btn')[idx].classList.add('active');
}


// Canvas
let scale = 1, pointX = 0, pointY = 0, panning = false, startX = 0, startY = 0;

async function initApp() {
    renderGapAnalysis();
    renderMegaFlow();
    renderDocs();
    renderStages();
    renderArchitectureMap();
    renderPlanNav();

    // Init Mermaid
    if (typeof mermaid !== 'undefined') {
        try {
            mermaid.initialize({ startOnLoad: false });
            await mermaid.run({ querySelector: '.mermaid' });
        } catch (e) { console.error("Mermaid error:", e); }
    }

    // Canvas Events
    const viewMega = document.getElementById('view-mega');
    viewMega.addEventListener('mousedown', (e) => {
        panning = true; startX = e.clientX - pointX; startY = e.clientY - pointY;
        viewMega.style.cursor = 'grabbing';
    });

    window.addEventListener('mouseup', () => {
        panning = false; viewMega.style.cursor = 'grab';
    });

    window.addEventListener('mousemove', (e) => {
        if (!panning) return;
        pointX = e.clientX - startX; pointY = e.clientY - startY;
        document.getElementById('canvas').style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
    });
}

function zoomCanvas(delta) {
    scale += delta; if (scale < 0.1) scale = 0.1;
    document.getElementById('canvas').style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
}

