window.onload = function() {
    drawAll();
    window.addEventListener('resize', drawAll);
};

function drawAll() {
    const svg = document.getElementById('svg-layer');
    const existingDefs = svg.querySelector('defs').outerHTML;
    svg.innerHTML = existingDefs;
    document.querySelectorAll('.connector-label, .cardinality').forEach(el => el.remove());

    /**
     * Draw dynamically computed connections strictly anchored to component edges
     */
    drawLine('FrontendClient', 'FastAPI_App', null, 'arrow', '-isteği başlatır', '0..*', '1');
    drawLine('FastAPI_App', 'OperationsRouter', 'aggregation', null, '-içerir', '1', '1..*');
    drawLine('FrontendClient', 'OperationsRouter', null, 'arrow', '-veri yollar', '1', '0..*');
    drawLine('OperationsRouter', 'ImageProcessingCore', null, 'arrow', '-işler', '1', '1..*');
    drawLine('OperationsRouter', 'DatabaseManager', null, 'arrow', '-loglar', '0..*', '1');
    drawLine('OperationsRouter', 'FileSystemStorage', null, 'arrow', '-kaydeder', '1', '0..*');
    drawLine('DatabaseManager', 'FileSystemStorage', null, 'arrow', '-okur/yazar', '1', '1');
    drawLine('ImageProcessingCore', 'BaseProcessingNode', null, 'inheritance', '', '', '');
}

function drawLine(id1, id2, startMarker, endMarker, labelText, textStart, textEnd) {
    const el1 = document.getElementById(id1);
    const el2 = document.getElementById(id2);
    if (!el1 || !el2) return;

    const canvasRect = document.getElementById('canvas').getBoundingClientRect();
    const r1 = el1.getBoundingClientRect();
    const r2 = el2.getBoundingClientRect();

    const c1 = { x: r1.left + r1.width/2 - canvasRect.left, y: r1.top + r1.height/2 - canvasRect.top };
    const c2 = { x: r2.left + r2.width/2 - canvasRect.left, y: r2.top + r2.height/2 - canvasRect.top };
    
    const b1 = { top: r1.top - canvasRect.top, bottom: r1.bottom - canvasRect.top, left: r1.left - canvasRect.left, right: r1.right - canvasRect.left, width: r1.width, height: r1.height };
    const b2 = { top: r2.top - canvasRect.top, bottom: r2.bottom - canvasRect.top, left: r2.left - canvasRect.left, right: r2.right - canvasRect.left, width: r2.width, height: r2.height };

    const dx = Math.abs(c1.x - c2.x);
    const dy = Math.abs(c1.y - c2.y);

    let p1, p2;

    // Detect if the dominant connection axis is horizontal or vertical
    if (dx > dy * 1.5) { 
        if (c1.x < c2.x) { p1 = { x: b1.right, y: c1.y }; p2 = { x: b2.left, y: c2.y }; }
        else { p1 = { x: b1.left, y: c1.y }; p2 = { x: b2.right, y: c2.y }; }
    } else {
        if (c1.y < c2.y) { p1 = { x: c1.x, y: b1.bottom }; p2 = { x: c2.x, y: b2.top }; }
        else { p1 = { x: c1.x, y: b1.top }; p2 = { x: c2.x, y: b2.bottom }; }
    }

    const svg = document.getElementById('svg-layer');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    
    let midX = (p1.x + p2.x) / 2;
    let midY = (p1.y + p2.y) / 2;
    let d = `M ${p1.x} ${p1.y} `;

    // Create orthogonal joint elbows protecting the whitespace gutters
    if (Math.abs(p1.x - p2.x) > 10 && Math.abs(p1.y - p2.y) > 10) {
        if (dx > dy * 1.5) { d += `L ${midX} ${p1.y} L ${midX} ${p2.y} L ${p2.x} ${p2.y}`; } 
        else { d += `L ${p1.x} ${midY} L ${p2.x} ${midY} L ${p2.x} ${p2.y}`; }
    } else {
        d += `L ${p2.x} ${p2.y}`;
    }

    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#34495e');
    path.setAttribute('stroke-width', '1.5');
    
    if (startMarker) path.setAttribute('marker-start', `url(#${startMarker})`);
    if (endMarker) path.setAttribute('marker-end', `url(#${endMarker})`);
    
    svg.appendChild(path);

    const canvas = document.getElementById('canvas');
    
    if (labelText) {
        const lbl = document.createElement('div');
        lbl.className = 'connector-label';
        lbl.innerText = labelText;
        lbl.style.left = midX + "px";
        lbl.style.top = midY + "px";
        canvas.appendChild(lbl);
    }
    
    if (textStart) {
        const cS = document.createElement('div');
        cS.className = 'cardinality';
        cS.innerText = textStart;
        if (dx > dy * 1.5) { 
            cS.style.left = (p1.x + (c1.x<c2.x?10:-30)) + "px"; 
            cS.style.top = (p1.y - 18) + "px"; 
        } else { 
            cS.style.left = (p1.x + 8) + "px"; 
            cS.style.top = (p1.y + (c1.y<c2.y?8:-25)) + "px"; 
        }
        canvas.appendChild(cS);
    }
    
    if (textEnd) {
        const cE = document.createElement('div');
        cE.className = 'cardinality';
        cE.innerText = textEnd;
        if (dx > dy * 1.5) { 
            cE.style.left = (p2.x + (c1.x<c2.x?-30:10)) + "px"; 
            cE.style.top = (p2.y - 18) + "px"; 
        } else { 
            cE.style.left = (p2.x + 8) + "px"; 
            cE.style.top = (p2.y + (c1.y<c2.y?-25:8)) + "px"; 
        }
        canvas.appendChild(cE);
    }
}

// Download Diagram as HD PNG Feature
document.addEventListener('DOMContentLoaded', function() {
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const btn = this;
            const originalText = btn.innerText;
            btn.innerText = "HD Görüntü İşleniyor...";
            btn.disabled = true;

            const node = document.getElementById('canvas');
            
            // Generate canvas using high pixel ratio for HD export
            htmlToImage.toPng(node, { 
                quality: 1.0, 
                pixelRatio: 2.5, // Enhances text crispness and SVG lines
                backgroundColor: '#ffffff',
                width: node.scrollWidth,
                height: node.scrollHeight + 10,
                style: {
                    transform: 'none',
                    margin: '0'
                }
            })
            .then(function (dataUrl) {
                let link = document.createElement('a');
                link.download = 'Görüntü_İşleme_UML_Mimarisi_HD.png';
                link.href = dataUrl;
                link.click();
                
                btn.innerText = originalText;
                btn.disabled = false;
            })
            .catch(function (error) {
                console.error('Export Error:', error);
                alert('PNG oluşturulurken bir hata oluştu! Geliştirici konsolunu kontrol edin.');
                btn.innerText = originalText;
                btn.disabled = false;
            });
        });
    }
});
