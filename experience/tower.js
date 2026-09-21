/* SLEEP scroll experience — scene 4: the transformer tower.
   Builds a 12-layer glass transformer (attention + MLP plates per layer),
   dismantles it, docks the five SLEEP organs, and reassembles it alive.
   Exposes: TOWER.build(scene), TOWER.setState(o), TOWER.setCam(camera,p4),
            TOWER.target(key). All classic-script, three.js r147. */
window.TOWER = (function(){
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
  const lerp=(a,b,t)=>a+(b-a)*t;

  const N=12, TOP=[8,9,10,11], MID=[4,5,6,7], SCALE=0.78, EXPLODE_K=1.05;
  const ATTN_H=0.055, MLP_H=0.11, GAP_IN=0.026, GAP_OUT=0.05;
  const FOOT_W=1.15, FOOT_D=0.78;

  let root=null, centerY=0, lastT=null, sceneRef=null;
  const layerMeshes=[];            // {attn, mlp}
  const pieces={};                 // ring, vault, valve, pod, wafers[]
  let pBlue=null, pGreen=null, blueMat=null, greenMat=null;
  let vaultBaseSpan=1, towerTopY=0;

  function edged(geo, color, edgeColor, opts){
    const m=new THREE.Mesh(geo, new THREE.MeshPhysicalMaterial(Object.assign({
      color, roughness:0.34, metalness:0.12, clearcoat:0.5, clearcoatRoughness:0.3,
      transparent:true, opacity:0.94 }, opts||{})));
    const e=new THREE.LineSegments(new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({color:edgeColor, transparent:true, opacity:0.5}));
    m.add(e); m.userData.edge=e.material;
    return m;
  }
  function glowSprite(hex, size){
    const c=document.createElement('canvas'); c.width=c.height=128;
    const g=c.getContext('2d');
    const gr=g.createRadialGradient(64,64,0,64,64,64);
    gr.addColorStop(0,hex+'ff'); gr.addColorStop(0.3,hex+'66'); gr.addColorStop(1,hex+'00');
    g.fillStyle=gr; g.fillRect(0,0,128,128);
    const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),
      transparent:true, blending:THREE.AdditiveBlending, depthWrite:false}));
    s.scale.setScalar(size); return s;
  }
  function particles(count, hex){
    const pos=new Float32Array(count*3), spd=new Float32Array(count);
    for(let i=0;i<count;i++){
      const a=Math.random()*Math.PI*2, r=Math.random()*0.13;
      pos[i*3]=Math.cos(a)*r; pos[i*3+1]=-1.7+Math.random()*3.6; pos[i*3+2]=Math.sin(a)*r;
      spd[i]=0.25+Math.random()*0.5;
    }
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
    const mat=new THREE.PointsMaterial({color:hex, size:0.035, transparent:true,
      opacity:0, blending:THREE.AdditiveBlending, depthWrite:false});
    const pts=new THREE.Points(geo,mat); pts.userData={spd};
    return [pts,mat];
  }

  function build(scene){
    sceneRef=scene;
    root=new THREE.Group(); root.scale.setScalar(SCALE); root.visible=false;
    scene.add(root);

    let y=-1.46;
    const base=edged(new THREE.BoxGeometry(1.3,0.09,0.92), 0x1c2338, 0x46527F);
    base.position.y=y+0.045; base.userData.cy=y+0.045; layerMeshes.push({solo:base});
    y+=0.09+GAP_OUT;

    for(let i=0;i<N;i++){
      const attn=edged(new THREE.BoxGeometry(FOOT_W,ATTN_H,FOOT_D), 0x2b3556, 0x5F74E8);
      attn.position.y=y+ATTN_H/2; attn.userData.cy=y+ATTN_H/2; y+=ATTN_H+GAP_IN;
      const mlp=edged(new THREE.BoxGeometry(FOOT_W,MLP_H,FOOT_D), 0x222b47, 0x3d4a78);
      mlp.position.y=y+MLP_H/2; mlp.userData.cy=y+MLP_H/2; y+=MLP_H+GAP_OUT;
      root.add(attn,mlp); layerMeshes.push({attn,mlp});
    }
    const head=edged(new THREE.BoxGeometry(0.9,0.09,0.65), 0x1c2338, 0x46527F);
    head.position.y=y+0.045; head.userData.cy=y+0.045; layerMeshes.push({solo:head});
    root.add(base,head);
    towerTopY=y+0.09;

    const cys=[]; layerMeshes.forEach(L=>{ if(L.solo)cys.push(L.solo.userData.cy);
      else cys.push(L.attn.userData.cy,L.mlp.userData.cy); });
    centerY=(Math.min(...cys)+Math.max(...cys))/2;

    /* --- the five organs (proposal-era placement) --- */
    // tagging ring above the head, around the token stream
    const ring=new THREE.Group();
    ring.add(edged(new THREE.TorusGeometry(0.42,0.024,16,64), 0x232a48, 0x8FA0FF,
      {opacity:0.95}));
    const ringGlow=glowSprite('#8FA0FF',0.55); ring.add(ringGlow);
    ring.rotation.x=Math.PI/2;
    ring.userData={to:new THREE.Vector3(0,towerTopY+0.30,0),
                   from:new THREE.Vector3(1.9,towerTopY+0.95,0), glow:ringGlow};
    root.add(ring); pieces.ring=ring;

    // KV vault beside the top-third attention plates
    const vault=new THREE.Group();
    const shell=edged(new THREE.BoxGeometry(0.26,1.0,0.6), 0x39406b, 0x8FA0FF,
      {opacity:0.55, clearcoat:1});
    vault.add(shell);
    for(let k=0;k<3;k++){
      const strip=new THREE.Mesh(new THREE.BoxGeometry(0.20,0.05,0.5),
        new THREE.MeshBasicMaterial({color:0x8FA0FF, transparent:true, opacity:0.8}));
      strip.position.y=-0.25+k*0.25; vault.add(strip);
    }
    vault.userData={to:new THREE.Vector3(0.95,0,0), from:new THREE.Vector3(2.4,0,0)};
    root.add(vault); pieces.vault=vault;

    // PRP valve: amber diamond with competing orbs
    const valve=new THREE.Group();
    valve.add(edged(new THREE.OctahedronGeometry(0.19), 0x2b2438, 0xE0A44A, {opacity:0.9}));
    valve.userData.orbs=[];
    for(let k=0;k<6;k++){
      const o=new THREE.Mesh(new THREE.SphereGeometry(0.034,12,12),
        new THREE.MeshBasicMaterial({color:0xE0A44A, transparent:true, opacity:0.85}));
      o.position.set(Math.cos(k*1.05)*0.16, 0.34+((k%3)*0.11), Math.sin(k*1.05)*0.12);
      o.userData.k=k; valve.add(o); valve.userData.orbs.push(o);
    }
    valve.userData.winners=[];
    for(let k=0;k<2;k++){
      const w=new THREE.Mesh(new THREE.SphereGeometry(0.04,12,12),
        new THREE.MeshBasicMaterial({color:0x4FB98A, transparent:true, opacity:0.95}));
      w.position.set(k?0.06:-0.06,-0.36,0); valve.add(w); valve.userData.winners.push(w);
    }
    valve.userData.to=new THREE.Vector3(0.95,-1.05,0);
    valve.userData.from=new THREE.Vector3(2.4,-1.05,0);
    root.add(valve); pieces.valve=valve;

    // sleep pod on the left: dark disc, green rim, crescent core
    const pod=new THREE.Group();
    const disc=edged(new THREE.CylinderGeometry(0.34,0.34,0.085,48), 0x141b2c, 0x4FB98A,
      {opacity:0.96});
    disc.rotation.x=Math.PI/2; pod.add(disc);
    const core=glowSprite('#4FB98A',0.34); core.position.z=0.06; pod.add(core);
    pod.userData={to:new THREE.Vector3(-1.05,0.05,0), from:new THREE.Vector3(-2.6,0.05,0),
                  glow:core};
    root.add(pod); pieces.pod=pod;

    // W_cons wafers -> proposal era: into the TOP-THIRD ATTENTION slots
    pieces.wafers=[];
    TOP.forEach((li,j)=>{
      const wf=new THREE.Mesh(new THREE.BoxGeometry(FOOT_W*0.98,0.016,FOOT_D*0.98),
        new THREE.MeshBasicMaterial({color:0x4FB98A, transparent:true, opacity:0}));
      const eg=new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(FOOT_W*0.98,0.016,FOOT_D*0.98)),
        new THREE.LineBasicMaterial({color:0x7FE8C0, transparent:true, opacity:0}));
      wf.add(eg); wf.userData={li, j, edge:eg.material};
      root.add(wf); pieces.wafers.push(wf);
    });

    // token streams
    [pBlue,blueMat]=particles(110, 0x9FB2FF);
    [pGreen,greenMat]=particles(110, 0x7FE8C0);
    root.add(pBlue,pGreen);

    // journey packet (scene 5): the fact travelling through the system
    pieces.packetB=glowSprite('#AFC0FF',0.30);
    pieces.packetG=glowSprite('#7FE8C0',0.30);
    pieces.packetB.material.opacity=0; pieces.packetG.material.opacity=0;
    root.add(pieces.packetB, pieces.packetG);

    // record vault base span against closed top-third positions
    const tys=TOP.map(li=>layerMeshes[li+1].attn.userData.cy);
    vaultBaseSpan=Math.max(...tys)-Math.min(...tys)+0.45;
    const sy=(Math.max(...tys)+Math.min(...tys))/2;
    vault.userData.to.y=sy; vault.userData.from.y=sy;
    shell.scale.y=vaultBaseSpan/1.0;

    // safety clamp: an amber wire cage around the wafer stack (scene 11)
    const cage=new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(FOOT_W*1.14,1,FOOT_D*1.18)),
      new THREE.LineBasicMaterial({color:0xE0A44A, transparent:true, opacity:0}));
    root.add(cage); pieces.clamp=cage;
  }

  /* o = { visible, explode, appear:{tag,kv,prp,engine,wafer}, alive, pulse:{...}, time } */
  function setState(o){
    if(!root) return;
    root.visible=o.visible;
    if(!o.visible) return;
    const t=o.time, dt=lastT===null?0.016:clamp(t-lastT,0,0.05); lastT=t;

    // explode layers
    layerMeshes.forEach(L=>{
      ['solo','attn','mlp'].forEach(kk=>{
        const m=L[kk]; if(!m) return;
        m.position.y=m.userData.cy+(m.userData.cy-centerY)*EXPLODE_K*o.explode;
      });
    });

    // ring rides the (possibly exploded) head plate
    const headPlate=layerMeshes[layerMeshes.length-1].solo;
    pieces.ring.userData.to.y=headPlate.position.y+0.42;
    pieces.ring.userData.from.y=headPlate.position.y+1.0;

    // organs: dock + pulse
    const dock=(g,tt)=>{
      const s=smooth(tt);
      g.position.lerpVectors(g.userData.from,g.userData.to,s);
      g.traverse(n=>{ if(n.material && n.material.transparent!==undefined && n.material.opacity!==undefined){
        if(n.userData.__baseOp===undefined) n.userData.__baseOp=n.material.opacity||1;
      }});
      g.traverse(n=>{ if(n.material&&n.userData.__baseOp!==undefined)
        n.material.opacity=n.userData.__baseOp*s; });
    };
    dock(pieces.ring,  o.appear.tag);
    dock(pieces.vault, o.appear.kv);
    dock(pieces.valve, o.appear.prp);
    dock(pieces.pod,   o.appear.engine);

    // ring spin + pulse
    pieces.ring.rotation.z=t*0.4;
    pieces.ring.userData.glow.material.opacity=
      o.appear.tag*(0.35+0.45*(0.5+0.5*Math.sin(t*3.2))*(0.3+0.7*(o.pulse.tag||0)));

    // vault tracks the (possibly exploded) top-third attention plates
    const ys=TOP.map(li=>layerMeshes[li+1].attn.position.y);
    const mid=(Math.max(...ys)+Math.min(...ys))/2;
    const span=Math.max(...ys)-Math.min(...ys)+0.45;
    pieces.vault.position.y=lerp(pieces.vault.userData.from.y,mid,smooth(o.appear.kv));
    pieces.vault.scale.y=span/vaultBaseSpan;

    // valve orbs churn; winners drop through when active
    const pw=o.pulse.prp||0;
    pieces.valve.rotation.y=t*0.5;
    pieces.valve.userData.orbs.forEach(orb=>{
      const k=orb.userData.k;
      orb.position.y=0.34+((k%3)*0.11)+Math.sin(t*2+k)*0.02;
      orb.material.opacity=o.appear.prp*(0.55+0.35*pw);
    });
    pieces.valve.userData.winners.forEach((w,k)=>{
      w.position.y=-0.36-0.10*pw*(0.5+0.5*Math.sin(t*2.4+k*2));
      w.material.opacity=o.appear.prp*(0.4+0.6*pw);
    });

    // pod breathes
    pieces.pod.rotation.z=t*0.25;
    pieces.pod.userData.glow.material.opacity=
      o.appear.engine*(0.3+0.5*(0.5+0.5*Math.sin(t*1.8))*(0.3+0.7*(o.pulse.engine||0)));

    // wafers: slide in (scene 4) or migrate top-attention -> mid-MLP (scene 11)
    const wm=o.waferMove||0;
    pieces.wafers.forEach((wf,idx)=>{
      const {li,j}=wf.userData;
      const topPlate=layerMeshes[li+1].attn;
      const midPlate=layerMeshes[MID[idx]+1].mlp;
      const yTop=topPlate.position.y+ATTN_H/2+0.014;
      const yMid=midPlate.position.y+MLP_H/2+0.014;
      const mt=clamp(wm*1.3-idx*0.09,0,1);
      if(mt<=0){
        const tj=clamp(o.appear.wafer*1.45-j*0.13,0,1), s=smooth(tj);
        wf.position.x=lerp(1.95+j*0.1,0,s);
        wf.position.y=yTop;
        wf.material.opacity=s*0.9; wf.userData.edge.opacity=s;
      } else {
        const p1=smooth(clamp(mt/0.32,0,1));
        const p2=smooth(clamp((mt-0.32)/0.42,0,1));
        const p3=smooth(clamp((mt-0.76)/0.24,0,1));
        wf.position.x=1.55*(p1-p3);
        wf.position.y=lerp(yTop,yMid,p2);
        wf.material.opacity=0.9; wf.userData.edge.opacity=1;
      }
    });

    // plate highlights (scene 11 explanations)
    const cAttnBase=0x5F74E8, cMlpBase=0x3d4a78, cAmber=0xE0A44A, cGreen=0x4FB98A;
    const _a=new THREE.Color(cAttnBase), _am=new THREE.Color(cAmber),
          _m=new THREE.Color(cMlpBase), _g=new THREE.Color(cGreen);
    TOP.forEach(li=>{
      const e=layerMeshes[li+1].attn.userData.edge;
      e.color.lerpColors(_a,_am,o.hiAttn||0);
      e.opacity=Math.max(e.opacity,(o.hiAttn||0)*(0.7+0.3*Math.sin(t*4)));
    });
    MID.forEach(li=>{
      const e=layerMeshes[li+1].mlp.userData.edge;
      e.color.lerpColors(_m,_g,o.hiMLP||0);
      e.opacity=Math.max(e.opacity,(o.hiMLP||0)*(0.7+0.3*Math.sin(t*4)));
    });

    // the safety clamp cage follows the wafer stack
    const con=o.clampOn||0;
    if(con>0){
      const ys=pieces.wafers.map(w=>w.position.y);
      const cy2=(Math.max(...ys)+Math.min(...ys))/2;
      const spanY=Math.max(...ys)-Math.min(...ys)+0.24;
      pieces.clamp.position.set(0,cy2,0);
      pieces.clamp.scale.y=spanY;
      const fix=o.clampFix||0;
      pieces.clamp.material.color.lerpColors(_am,_g,fix);
      const grip=1-0.06*Math.sin(t*3)*(1-fix);
      pieces.clamp.scale.x=grip+0.10*fix;
      pieces.clamp.scale.z=grip+0.10*fix;
      pieces.clamp.material.opacity=con*(0.7+0.3*Math.sin(t*2.5));
    } else pieces.clamp.material.opacity=0;

    // token streams
    const adv=(pts)=>{
      const a=pts.geometry.attributes.position, spd=pts.userData.spd;
      for(let i=0;i<spd.length;i++){
        let yy=a.getY(i)+spd[i]*dt*(0.4+0.6*(1-o.explode));
        if(yy>1.95) yy=-1.7;
        a.setY(i,yy);
      }
      a.needsUpdate=true;
    };
    adv(pBlue); adv(pGreen);
    blueMat.opacity=0.75*(1-o.explode)*(1-o.alive*0.85);
    greenMat.opacity=0.85*o.alive*(1-o.explode);

    // reassembled = alive: edges brighten
    const eb=0.5+0.4*o.alive;
    layerMeshes.forEach(L=>['solo','attn','mlp'].forEach(kk=>{
      const m=L[kk]; if(m) m.userData.edge.opacity=eb;
    }));
  }

  /* camera: [key p4, radius, azimuth, polar, target y (local)] */
  const CAMS=[
    [0.030, 8.6, -0.35, 1.30, 0.1],
    [0.115, 5.8, -0.40, 1.22, 0.1],
    [0.205, 7.6, -0.12, 1.15, 0.15],
    [0.315, 5.8, -0.05, 0.96, 2.65],
    [0.445, 5.4,  0.62, 1.10, 1.35],
    [0.575, 5.2,  0.55, 1.44, -1.55],
    [0.705, 5.4,  2.55, 1.22, 0.10],
    [0.825, 4.8,  0.38, 1.05, 1.55],
    [0.935, 7.4, -0.55, 1.18, 0.10],
  ];
  function setCam(camera,p4){
    let i=0;
    while(i<CAMS.length-2 && p4>CAMS[i+1][0]) i++;
    const A=CAMS[i], B=CAMS[i+1];
    const f=smooth((p4-A[0])/(B[0]-A[0]));
    const r=lerp(A[1],B[1],f), az=lerp(A[2],B[2],f),
          po=lerp(A[3],B[3],f), ty=lerp(A[4],B[4],f)*SCALE;
    camera.position.set(r*Math.sin(po)*Math.cos(az), r*Math.cos(po)+ty,
                        r*Math.sin(po)*Math.sin(az));
    camera.lookAt(0,ty,0);
  }

  function target(key){
    return {tag:pieces.ring, kv:pieces.vault, prp:pieces.valve,
            engine:pieces.pod, wafer:pieces.wafers&&pieces.wafers[3],
            base:layerMeshes[0]&&layerMeshes[0].solo,
            mid:layerMeshes[6]&&layerMeshes[6].mlp,
            clamp:pieces.clamp}[key]||null;
  }

  /* camera for scene 11 — the repair */
  const RCAMS=[
    [0.000, 7.6, -0.45, 1.20, 0.10],
    [0.055, 6.0, -0.35, 1.18, 0.10],   // the tower returns
    [0.135, 7.4, -0.10, 1.12, 0.15],   // open it up
    [0.225, 4.9,  0.35, 1.02, 1.55],   // the wrong address (top third)
    [0.355, 4.7,  0.30, 1.30, -0.45],  // the right address (mid MLPs)
    [0.505, 6.2,  0.45, 1.16, 0.55],   // the move (wide, follow)
    [0.650, 4.4,  0.30, 1.28, -0.45],  // the clip, gripping the wafers
    [0.770, 4.6,  0.25, 1.26, -0.45],  // right-sizing it
    [0.880, 6.6, -0.35, 1.14, 0.10],   // reassembly
    [0.980, 7.8, -0.55, 1.20, 0.10],
  ];
  function setCamRepair(camera,p){
    let i=0;
    while(i<RCAMS.length-2 && p>RCAMS[i+1][0]) i++;
    const A=RCAMS[i], B=RCAMS[i+1];
    const f=smooth((p-A[0])/(B[0]-A[0]));
    const r=lerp(A[1],B[1],f), az=lerp(A[2],B[2],f),
          po=lerp(A[3],B[3],f), ty=lerp(A[4],B[4],f)*SCALE;
    camera.position.set(r*Math.sin(po)*Math.cos(az), r*Math.cos(po)+ty,
                        r*Math.sin(po)*Math.sin(az));
    camera.lookAt(0,ty,0);
  }

  /* ---------- scene 5: one fact's journey ----------
     jPos runs 0..7 across waypoints; ws carries per-stage weights. */
  function journeyWaypoints(){
    const headY=layerMeshes[layerMeshes.length-1].solo.position.y;
    const ringY=headY+0.42;
    const tys=TOP.map(li=>layerMeshes[li+1].attn.position.y);
    const topMid=(Math.max(...tys)+Math.min(...tys))/2;
    return [
      new THREE.Vector3(0,-2.9,0),          // 0 below the tower
      new THREE.Vector3(0, 0.0,0),          // 1 inside, mid climb
      new THREE.Vector3(0, ringY,0),        // 2 at the ring (read)
      new THREE.Vector3(0, ringY,0),        // 3 tag fires (hold)
      new THREE.Vector3(0.95, topMid,0),    // 4 into the vault
      new THREE.Vector3(0.95,-1.05,0),      // 5 down to the valve
      new THREE.Vector3(-1.05,0.05,0),      // 6 across to the pod
      new THREE.Vector3(0, topMid,0),       // 7 into the wafers / out
    ];
  }
  function setJourney(o){
    const bOp=pieces.packetB.material, gOp=pieces.packetG.material;
    if(!o.active){ bOp.opacity=0; gOp.opacity=0; return; }
    const W=journeyWaypoints();
    const i=clamp(Math.floor(o.jPos),0,W.length-2);
    const f=smooth(o.jPos-i);
    const pos=new THREE.Vector3().lerpVectors(W[i],W[i+1],f);
    if(i===3||i===4||i===5||i===6) pos.y+=Math.sin(f*Math.PI)*0.28;  // arcs
    const pulse=1+0.18*Math.sin(o.time*5);
    [pieces.packetB,pieces.packetG].forEach(s=>{
      s.position.copy(pos); s.scale.setScalar(0.30*pulse);
    });
    const greenMix=smooth((o.jPos-5.4)/0.8);   // turns green through the pod
    const vis=smooth(o.jPos/0.4)*(1-smooth((o.jPos-6.55)/0.45));
    bOp.opacity=vis*(1-greenMix); gOp.opacity=vis*greenMix;

    // organ side-effects
    const ws=o.ws;
    pieces.ring.userData.glow.material.opacity=
      Math.max(pieces.ring.userData.glow.material.opacity,
               0.5*(ws.read||0)+0.85*(ws.tag||0));
    pieces.pod.userData.glow.material.opacity=
      Math.max(pieces.pod.userData.glow.material.opacity, 0.9*(ws.sleep||0));
    // wafers glow as consolidation happens, then stay warm
    const wGlow=0.55*(ws.sleep||0)+0.9*(ws.out||0);
    pieces.wafers.forEach((wf,j)=>{
      const tw=0.5+0.5*Math.sin(o.time*3+j*1.3);
      wf.material.opacity=Math.max(wf.material.opacity, (0.5+0.5*tw)*wGlow*0.9);
    });
    // the answer flows: green tokens rise at the end
    greenMat.opacity=Math.max(greenMat.opacity, 0.9*(ws.out||0));
    blueMat.opacity=Math.min(blueMat.opacity, 0.75*(1-0.8*(ws.out||0)));
  }

  const JCAMS=[
    [0.000, 7.4, -0.55, 1.18,  0.1],
    [0.075, 6.2, -0.45, 1.35, -1.2],   // input, low
    [0.210, 5.6, -0.25, 1.02,  2.0],   // read, rise to the ring
    [0.345, 4.9, -0.10, 0.98,  2.3],   // tag
    [0.475, 5.0,  0.55, 1.10,  1.35],  // vault
    [0.605, 5.0,  0.55, 1.42, -1.35],  // valve
    [0.740, 5.2,  2.55, 1.22,  0.1],   // pod
    [0.890, 5.4,  0.25, 1.02,  1.5],   // wafers / out
    [0.985, 7.0, -0.45, 1.15,  0.2],   // settle
  ];
  function setCamJourney(camera,p5){
    let i=0;
    while(i<JCAMS.length-2 && p5>JCAMS[i+1][0]) i++;
    const A=JCAMS[i], B=JCAMS[i+1];
    const f=smooth((p5-A[0])/(B[0]-A[0]));
    const r=lerp(A[1],B[1],f), az=lerp(A[2],B[2],f),
          po=lerp(A[3],B[3],f), ty=lerp(A[4],B[4],f)*SCALE;
    camera.position.set(r*Math.sin(po)*Math.cos(az), r*Math.cos(po)+ty,
                        r*Math.sin(po)*Math.sin(az));
    camera.lookAt(0,ty,0);
  }

  /* ================= scene 15: the memory shelf ================= */
  let shelfRoot=null, shelfCubes=[], shelfSweep=null;
  const C_DIM=new THREE.Color(0x152218), C_BRT=new THREE.Color(0x4FB98A);
  function buildShelf(){
    shelfRoot=new THREE.Group(); shelfRoot.visible=false; sceneRef.add(shelfRoot);
    for(let i=0;i<10;i++){
      const m=edged(new THREE.BoxGeometry(0.36,0.36,0.36), 0x152218, 0x2C7D57,
        {opacity:0.96});
      m.position.set(i*0.5-2.25, 0, 0);
      shelfRoot.add(m); shelfCubes.push(m);
    }
    shelfSweep=glowSprite('#7FE8C0',0.9); shelfSweep.material.opacity=0;
    shelfRoot.add(shelfSweep);
  }
  /* o = {visible, appearT, fadeAmt, sweepX(null|x), time} */
  function setShelf(o){
    if(!shelfRoot) return;
    shelfRoot.visible=o.visible; if(!o.visible) return;
    shelfCubes.forEach((m,i)=>{
      const appeared=smooth(clamp(o.appearT*11-i,0,1));
      const age=clamp((o.appearT*11-i-1)/9,0,1);
      let bright=appeared*(1-0.78*o.fadeAmt*age);
      if(o.sweepX!==null){
        const relight=smooth((o.sweepX-m.position.x+0.35)/0.7);
        bright=Math.max(bright, appeared*(0.25+0.72*relight));
        const d=Math.abs(m.position.x-o.sweepX);
        bright=Math.min(1, bright+Math.exp(-d*d*3)*0.5);
      }
      m.material.color.lerpColors(C_DIM,C_BRT,bright);
      m.userData.edge.opacity=0.25+0.75*bright;
      const s=0.92+0.14*bright+0.02*Math.sin(o.time*2+i);
      m.scale.setScalar(s);
    });
    if(o.sweepX!==null){
      shelfSweep.position.set(o.sweepX, 0.42, 0.1);
      shelfSweep.material.opacity=0.85;
    } else shelfSweep.material.opacity=0;
  }
  function setCamShelf(camera,p){
    const f=smooth(p);
    const r=lerp(5.6,4.9,f), az=lerp(-0.55,0.40,f), po=lerp(1.22,1.12,f);
    camera.position.set(r*Math.sin(po)*Math.cos(az), r*Math.cos(po),
                        r*Math.sin(po)*Math.sin(az));
    camera.lookAt(0,0,0);
  }

  /* ================= scene 16: two offices, one memory ================= */
  let officeRoot=null; const off={};
  function person(shirtHex, skinHex){
    const g=new THREE.Group();
    const body=new THREE.Mesh(new THREE.CapsuleGeometry(0.17,0.34,6,14),
      new THREE.MeshPhysicalMaterial({color:shirtHex, roughness:0.6}));
    body.position.y=0.62; g.add(body);
    const head=new THREE.Mesh(new THREE.SphereGeometry(0.145,20,16),
      new THREE.MeshPhysicalMaterial({color:skinHex, roughness:0.55}));
    head.position.y=1.05; g.add(head);
    g.userData.head=head;
    return g;
  }
  function deskSet(sx, shirt, skin){
    const g=new THREE.Group();
    const desk=edged(new THREE.BoxGeometry(1.15,0.06,0.62), 0x1a2133, 0x46527F);
    desk.position.set(0,0.58,0); g.add(desk);
    const mon=new THREE.Group();
    const frame=edged(new THREE.BoxGeometry(0.52,0.36,0.045), 0x10141f, 0x46527F);
    mon.add(frame);
    const scr=new THREE.Mesh(new THREE.PlaneGeometry(0.46,0.30),
      new THREE.MeshBasicMaterial({color:0x22305a, transparent:true, opacity:0.95}));
    scr.position.z=0.026; mon.add(scr);
    mon.position.set(0,0.86,0.05);
    mon.rotation.y=sx>0?0.5:-0.5;
    g.add(mon);
    const p=person(shirt, skin);
    p.position.set(sx>0?0.35:-0.35, 0, 0.55);
    g.add(p);
    g.position.set(sx,0,0.2);
    return {g, screen:scr, person:p, monitor:mon};
  }
  function buildOffice(){
    officeRoot=new THREE.Group(); officeRoot.visible=false; sceneRef.add(officeRoot);
    const ground=new THREE.Mesh(new THREE.CircleGeometry(7.5,56),
      new THREE.MeshPhysicalMaterial({color:0x0d1119, roughness:1}));
    ground.rotation.x=-Math.PI/2; ground.position.y=-0.01; officeRoot.add(ground);

    // the shared SLEEP model on a pedestal, wafers glowing mid-stack
    const mini=new THREE.Group();
    const ped=edged(new THREE.BoxGeometry(0.8,0.5,0.8), 0x141b2c, 0x46527F);
    ped.position.y=0.25; mini.add(ped);
    let my=0.55;
    for(let i=0;i<6;i++){
      const sl=edged(new THREE.BoxGeometry(0.55,0.075,0.4),
        i%2?0x222b47:0x2b3556, i%2?0x3d4a78:0x5F74E8, {opacity:0.95});
      sl.position.y=my+0.038; mini.add(sl); my+=0.105;
      if(i===2||i===3){
        const wfm=new THREE.Mesh(new THREE.BoxGeometry(0.53,0.014,0.38),
          new THREE.MeshBasicMaterial({color:0x4FB98A, transparent:true, opacity:0.9}));
        wfm.position.y=my-0.018; mini.add(wfm);
      }
    }
    const halo=glowSprite('#8FA0FF',0.5); halo.position.y=my+0.25; mini.add(halo);
    off.halo=halo; off.miniTop=halo;
    officeRoot.add(mini); off.mini=mini;

    const A=deskSet(-2.3, 0x3D52C9, 0xE8C39E);
    const B=deskSet( 2.3, 0x2C7D57, 0x8a5a33);
    officeRoot.add(A.g,B.g); off.A=A; off.B=B;

    off.doc=new THREE.Mesh(new THREE.BoxGeometry(0.14,0.18,0.02),
      new THREE.MeshBasicMaterial({color:0xE6EAF4, transparent:true, opacity:0}));
    officeRoot.add(off.doc);
    off.ans=glowSprite('#7FE8C0',0.3); off.ans.material.opacity=0; officeRoot.add(off.ans);
    off.moon=glowSprite('#AFC0FF',0.55); off.moon.position.set(0,2.1,0);
    off.moon.material.opacity=0; officeRoot.add(off.moon);
  }
  /* o={visible,time,docT,night,speakA,ansTa,speakB,ansTb} */
  function setOffice(o){
    if(!officeRoot) return;
    officeRoot.visible=o.visible; if(!o.visible) return;
    const t=o.time;
    [[off.A,0],[off.B,2.1]].forEach(([D,ph])=>{
      D.person.position.y=0.015*Math.sin(t*1.5+ph);
      D.person.userData.head.rotation.z=0.08*Math.sin(t*0.8+ph);
    });
    off.halo.material.opacity=0.3+0.25*Math.sin(t*2)+0.3*(o.night||0);
    // document flies A's screen -> model
    const dT=o.docT||0;
    if(dT>0&&dT<1){
      const a=new THREE.Vector3(-1.9,1.0,0.25), b=new THREE.Vector3(0,1.35,0);
      off.doc.position.lerpVectors(a,b,smooth(dT));
      off.doc.position.y+=Math.sin(dT*Math.PI)*0.5;
      off.doc.rotation.y=dT*4;
      off.doc.material.opacity=Math.sin(dT*Math.PI);
    } else off.doc.material.opacity=0;
    off.moon.material.opacity=(o.night||0)*(0.7+0.2*Math.sin(t*1.5));
    // answers: green pulse model -> screen
    const ans=(tt,sx)=>{
      if(tt<=0||tt>=1){ return; }
      const a=new THREE.Vector3(0,1.35,0), b=new THREE.Vector3(sx,1.0,0.25);
      off.ans.position.lerpVectors(a,b,smooth(tt));
      off.ans.material.opacity=Math.sin(tt*Math.PI);
    };
    off.ans.material.opacity=0;
    ans(o.ansTa||0,-1.9); ans(o.ansTb||0, 1.9);
    off.A.screen.material.color.setHex((o.speakA||0)>0.5?0x2C7D57:0x22305a);
    off.B.screen.material.color.setHex((o.speakB||0)>0.5?0x2C7D57:0x22305a);
  }
  const OCAMS=[
    [0.00, 4.6, -2.40, 1.18, -1.9],
    [0.30, 4.4, -2.30, 1.15, -1.6],
    [0.46, 5.6, -1.57, 1.10,  0.0],   // pan across the model
    [0.58, 4.4, -0.85, 1.15,  1.6],
    [0.78, 4.6, -0.70, 1.18,  1.9],
    [0.92, 7.2, -1.57, 1.08,  0.0],   // pull wide: both offices, one memory
  ];
  function setCamOffice(camera,p){
    let i=0;
    while(i<OCAMS.length-2 && p>OCAMS[i+1][0]) i++;
    const A=OCAMS[i], B=OCAMS[i+1];
    const f=smooth((p-A[0])/(B[0]-A[0]));
    const r=lerp(A[1],B[1],f), az=lerp(A[2],B[2],f), po=lerp(A[3],B[3],f),
          tx=lerp(A[4],B[4],f);
    camera.position.set(tx*0.4+r*Math.sin(po)*Math.cos(az), r*Math.cos(po)+0.8,
                        r*Math.sin(po)*Math.sin(az));
    camera.lookAt(tx,0.8,0);
  }
  function buildExtras(){ buildShelf(); buildOffice(); }
  function xtarget(key){
    return { cubeOld:shelfCubes[0], cubeNew:shelfCubes[9], sweep:shelfSweep,
             headA:off.A&&off.A.person.userData.head, screenA:off.A&&off.A.monitor,
             headB:off.B&&off.B.person.userData.head, screenB:off.B&&off.B.monitor,
             mini:off.miniTop }[key]||null;
  }

  return { build, setState, setCam, target, setJourney, setCamJourney, setCamRepair,
           buildExtras, setShelf, setCamShelf, setOffice, setCamOffice, xtarget };
})();
