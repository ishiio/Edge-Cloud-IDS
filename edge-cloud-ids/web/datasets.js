const snapshot = window.IDS_SNAPSHOT;
const catalog = snapshot.catalog;
let selectedDataset = Object.keys(catalog)[0], previewPage = 0;
const number = n => Number(n).toLocaleString();
const PALETTE=['#ff6b86','#ffb454','#ffd75e','#a384ff','#4fd8ff','#ff6ec7','#5b9dff','#b6e354','#ff8a5b','#7ee0a8','#d38bff','#5fe0e0','#f5a3c7','#9ab0ff','#e6c36a'];
const classColor=(d,label)=>{if(label===d.benign)return '#2de2c0';const ks=Object.keys(d.class_counts).filter(k=>k!==d.benign);return PALETTE[Math.max(0,ks.indexOf(label))%PALETTE.length];};
const copyByTab={datasets:['Your data, in focus.','Two real-world datasets. Every flow, feature, and attack class within reach.'],monitor:['See every decision.','Replay held-out traffic through the edge-to-cloud inference pipeline.'],lab:['Find your balance.','Explore detection quality, cloud traffic, and latency at each threshold.'],bench:['Evidence, at a glance.','Compare saved test-set results from the real datasets.'],analyze:['Look closer at a flow.','Classify your records with the trained edge and cloud models.'],arch:['Intelligence, connected.','Follow the path from a recorded flow to its final decision.']};
function setPageHeading(key){const t=copyByTab[key];if(!t)return;$('#workspaceTitle').textContent=t[0];$('#workspaceDescription').textContent=t[1];$('#crumb').textContent=$(`#tabs button[data-tab="${key}"] span`).textContent;$$('#tabs button').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===key)));}
$$('#tabs button').forEach(b=>b.addEventListener('click',()=>setPageHeading(b.dataset.tab)));
$('.brand').onclick=e=>{e.preventDefault();$('#tabs button[data-tab="datasets"]').click();};
function openGuide(){$('#startupGuide').showModal();}
$('#openGuide').onclick=openGuide;$('#closeGuide').onclick=()=>$('#startupGuide').close();
$('#retryConnection').onclick=()=>{if(location.protocol==='file:')location.href='http://127.0.0.1:8000/#datasets';else location.reload();};
function renderDatasets(){
 const d=catalog[selectedDataset];
 $('#totalRows').textContent=number(Object.values(catalog).reduce((s,d)=>s+d.rows,0));
 $('#totalClasses').textContent=number(Object.values(catalog).reduce((s,d)=>s+Object.keys(d.class_counts).length,0));
 $('#datasetCards').innerHTML=Object.values(catalog).map((c,i)=>`<button class="dataset-card ${c.key===selectedDataset?'selected':''}" data-dataset="${c.key}" aria-pressed="${c.key===selectedDataset}"><div class="dataset-card-top"><span class="dataset-symbol ${i?'violet':''}">${i?'TN':'E²'}</span><span class="available"><i></i> CSV included</span><span class="select-dot">${c.key===selectedDataset?'✓':''}</span></div><h2>${esc(c.title)}</h2><p>${i?'Network telemetry · UNSW TON_IoT':'Industrial IoT · Edge-IIoTset ML'}</p><div class="dataset-card-bottom"><span><b>${number(c.rows)}</b> raw flows</span><span><b>${Object.keys(c.class_counts).length}</b> classes</span><span class="arrow">↗</span></div></button>`).join('');
 $$('[data-dataset]').forEach(b=>b.onclick=()=>{selectedDataset=b.dataset.dataset;previewPage=0;$('#previewSearch').value='';renderDatasets();});
 $('#explorerTitle').textContent=d.title;
 $('#downloadDataset').href=location.protocol==='file:'?d.path:`/api/datasets/${d.key}/download`;
 $('#downloadDataset').setAttribute('download',d.filename);
 $('#datasetFacts').innerHTML=[['After deduplication',number(d.unique_rows),'unique flows'],['CSV columns',d.columns.length,'columns in CSV'],['Policy A macro-F1',d.policy_a_f1.toFixed(2)+'%','test set · seed 0'],['Cloud forwarding',d.forwarded.toFixed(1)+'%','Policy A · selected threshold']].map(([l,v,s])=>`<div><span>${l}</span><b>${v}</b><small>${s}</small></div>`).join('');
 const benign=d.class_counts[d.benign], attacks=d.rows-benign, attackPct=100*attacks/d.rows;
 $('#distributionSummary').innerHTML=`<div class="composition-overview"><div class="donut" style="--attack:${attackPct}%"><div><b>${attackPct.toFixed(1)}<small>%</small></b><span>attack records</span></div></div><div class="composition-legend"><span><i class="dot cloud"></i>Attack <b>${number(attacks)}</b></span><span><i class="dot edge"></i>Benign <b>${number(benign)}</b></span></div></div>`;
 const mx=Math.max(...Object.values(d.class_counts));
 $('#distribution').innerHTML=Object.entries(d.class_counts).map(([label,n])=>`<button class="class-row" style="--cc:${classColor(d,label)}" data-class="${esc(label)}" title="Filter preview: ${esc(label)}"><span>${esc(label.replaceAll('_',' '))}</span><b>${number(n)}</b><div><i style="width:${100*n/mx}%;background:${classColor(d,label)}"></i></div></button>`).join('');
 $('#previewClass').innerHTML='<option value="">All classes</option>'+Object.keys(d.class_counts).map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('');
 $$('[data-class]').forEach(b=>b.onclick=()=>{$('#previewClass').value=b.dataset.class;previewPage=0;renderPreview();});
 $('#previewNote').textContent=`Representative preview · up to 5 source rows per class · ${d.columns.length} columns available`;
 $('#dictionary').innerHTML='<thead><tr><th>Feature</th><th>Data type</th><th>Missing values</th></tr></thead><tbody>'+d.fields.map(f=>`<tr><td>${esc(f.name)}</td><td><span class="tag">${esc(f.dtype)}</span></td><td>${number(f.missing)}</td></tr>`).join('')+'</tbody>';
 $('#datasetProvenance').innerHTML=`<span>↳ ${esc(d.filename)} · ${(d.bytes/1e6).toFixed(1)} MB · user-supplied CSV</span><details><summary>File integrity & evaluation notes</summary><p>SHA-256 <code>${d.sha256}</code></p><p>${number(d.duplicates)} exact duplicate rows removed before splitting. Model results use seed 0 with a stratified test split; these are independent implementation results, not a verified reproduction of the paper. This view is generated from the bundled files.</p></details>`;
 renderPreview();
}
function renderPreview(){const d=catalog[selectedDataset],q=$('#previewSearch').value.toLowerCase(),cl=$('#previewClass').value;
 const rows=d.preview.filter(r=>(!cl||r[d.label]===cl)&&(!q||Object.values(r).some(v=>String(v??'').toLowerCase().includes(q))));const pages=Math.max(1,Math.ceil(rows.length/10));previewPage=Math.min(previewPage,pages-1);
 const cols=$('#allColumns').checked?d.columns:d.columns.slice(0,7);
 $('#datasetTable').innerHTML='<thead><tr>'+cols.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr></thead><tbody>'+rows.slice(previewPage*10,previewPage*10+10).map(r=>'<tr>'+cols.map(c=>`<td title="${esc(r[c])}">${c===d.label?`<span class="tag ${r[c]===d.benign?'ok':'bad'}" style="--tc:${classColor(d,r[c])}">${esc(r[c])}</span>`:esc(r[c]??'—')}</td>`).join('')+'</tr>').join('')+(!rows.length?`<tr><td colspan="${cols.length}" class="empty">No preview rows match this filter.</td></tr>`:'')+'</tbody>';
 $('#previewCount').textContent=`${rows.length?previewPage*10+1:0}–${Math.min((previewPage+1)*10,rows.length)} of ${rows.length} preview rows`;
 $('#previewPage').textContent=` ${previewPage+1} / ${pages} `;$('#prevRows').disabled=previewPage===0;$('#nextRows').disabled=previewPage===pages-1;
}
$('#previewSearch').oninput=()=>{previewPage=0;renderPreview();};$('#previewClass').onchange=()=>{previewPage=0;renderPreview();};$('#allColumns').onchange=renderPreview;
$('#prevRows').onclick=()=>{previewPage--;renderPreview();};$('#nextRows').onclick=()=>{previewPage++;renderPreview();};
$('#useDataset').onclick=async()=>{if(offlineMode){openGuide();return;}await pushConfig({dataset:selectedDataset});$('#tabs button[data-tab="monitor"]').click();};
renderDatasets();
