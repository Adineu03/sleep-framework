/* SLEEP scroll experience — scene 4: the transformer tower.
   Builds a 12-layer glass transformer (attention + MLP plates per layer),
   dismantles it, docks the five SLEEP organs, and reassembles it alive.
   Exposes: TOWER.build(scene), TOWER.setState(o), TOWER.setCam(camera,p4),
            TOWER.target(key). All classic-script, three.js r147. */
window.TOWER = (function(){
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
  const lerp=(a,b,t)=>a+(b-a)*t;

  const N=12, TOP=[8,9,10,11], SCALE=0.78, EXPLODE_K=1.05;
  const ATTN_H=0.055, MLP_H=0.11, GAP_IN=0.026, GAP_OUT=0.05;
  const FOOT_W=1.15, FOOT_D=0.78;

  let root=null, centerY=0, lastT=null;
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

    // record vault base span against closed top-third positions
    const tys=TOP.map(li=>layerMeshes[li+1].attn.userData.cy);
    vaultBaseSpan=Math.max(...tys)-Math.min(...tys)+0.45;
    const sy=(Math.max(...tys)+Math.min(...tys))/2;
    vault.userData.to.y=sy; vault.userData.from.y=sy;
    shell.scale.y=vaultBaseSpan/1.0;
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

    // wafers ride their attention plates; slide in on the x axis
    pieces.wafers.forEach(wf=>{
      const {li,j}=wf.userData;
      const plate=layerMeshes[li+1].attn;
      wf.position.y=plate.position.y+ATTN_H/2+0.014;
      const tj=clamp(o.appear.wafer*1.45-j*0.13,0,1), s=smooth(tj);
      wf.position.x=lerp(1.95+j*0.1,0,s);
      wf.material.opacity=s*0.9; wf.userData.edge.opacity=s;
    });

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
            engine:pieces.pod, wafer:pieces.wafers&&pieces.wafers[3]}[key]||null;
  }

  return { build, setState, setCam, target };
})();
