'use strict';
const figures={
 quiver:{file:'quiver_overlay',title:'Conservative velocity fields',caption:'Both methods share arrow origins and scale. Arrows show four times the displacement between frames; missing estimates are omitted. The cyan line is the local surface and the orange line is 1 cm below it.',alt:'Full-width quiver comparison for pairs 80 and 100: orange optical flow and blue PIV, with a 5 cm/s arrow key.'},
 mean:{file:'mean_horizontal_velocity_with_surface',title:'Mean horizontal velocity with IR surface references',caption:'The two subsurface means use the same horizontal segments at each depth. Gold stars mark the smoothed IR observations at zero depth: 9.87 and 13.49 cm/s. The unavailable near-surface band is left empty.',alt:'Mean horizontal velocity versus local depth for pairs 80 and 100, comparing orange optical flow, dashed blue PIV, and gold IR stars at the surface.'},
 du:{file:'du_dx_vs_piv',title:'Horizontal gradient ∂u/∂x',caption:'Analytic optical-flow derivatives, PIV central differences, and their overlap differences. These are Cartesian gradients displayed at depth below the local surface; the methods retain different smoothing. Missing cells stay blank.',alt:'Horizontal velocity gradient maps for optical flow, PIV and differences in pairs 80 and 100.'},
 dw:{file:'dw_dz_vs_piv',title:'Vertical gradient ∂w/∂z',caption:'Vertical velocity and physical height are both positive upward. The two image-to-physical sign changes cancel in this derivative. Gradient screening is stricter than vector screening; blank regions are unsupported.',alt:'Vertical velocity gradient maps for optical flow, PIV and differences in pairs 80 and 100.'},
 integral:{file:'horizontal_integral_vs_piv_with_surface',title:'Horizontal integrals, mean velocities, and coverage',caption:'Integrate horizontally along a contour of constant vertical depth below the local surface. Both methods use the same accepted intervals at each depth. The coverage panel shows how much width contributes; no full-width value is invented.',alt:'Horizontal integral, mean horizontal velocity with IR star, and shared coverage versus depth for two image pairs.'},
 coverage:{file:'horizontal_segments_used',title:'The horizontal segments used in the comparisons',caption:'Shared support can change with depth. A fixed common domain is the intersection of valid intervals over every depth in the selected band; that intersection is empty for these examples. Per-depth integrals remain available on their reported segments.',alt:'Maps of horizontally shared valid integration segments as a function of local depth for pairs 80 and 100.'}
};
const resultImage=document.getElementById('result-image');
document.querySelectorAll('[data-figure]').forEach(button=>button.addEventListener('click',()=>{
 const f=figures[button.dataset.figure];
 document.querySelectorAll('[data-figure]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 resultImage.src='assets/'+f.file+'.png';resultImage.alt=f.alt;resultImage.removeAttribute('width');resultImage.removeAttribute('height');
 document.getElementById('figure-link').href='assets/'+f.file+'.svg';
 document.getElementById('figure-download').href='assets/'+f.file+'.svg';
 document.getElementById('figure-title').textContent=f.title;
 document.getElementById('figure-caption').textContent=f.caption;
}));
document.getElementById('copy-command').addEventListener('click',async()=>{
 const status=document.getElementById('copy-status');
 try{await navigator.clipboard.writeText(document.getElementById('run-command').textContent);status.textContent='Commands copied.';}
 catch(error){const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(document.getElementById('run-command'));selection.removeAllRanges();selection.addRange(range);status.textContent='Commands selected. Use your keyboard copy shortcut.';}
});
