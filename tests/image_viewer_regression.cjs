const fs=require('fs');const vm=require('vm');const assert=require('assert');
const documents=JSON.parse(fs.readFileSync(0,'utf8'));
const script=documents[0].match(/<script>([\s\S]*?)<\/script>/)[1];
function scenario(naturalWidth, naturalHeight, code = script) {
  const handlers = new Map();
  let box = { width:400, height:700, left:20, top:10 };
  let resize;
  const elements = {};
  for (const id of ['viewer', 'image', 'zoom', 'reset', 'status']) {
    const classes = new Set();
    const captures = new Set();
    elements[id] = {
      style:{}, hidden:false, complete:true, naturalWidth, naturalHeight,
      addEventListener(name, handler, options) {handlers.set(id + ':' + name, {handler, options});},
      getBoundingClientRect() {return box;},
      focus() {},
      classList:{add(c){classes.add(c);},remove(c){classes.delete(c);},contains(c){return classes.has(c);}},
      setPointerCapture(id){captures.add(id);}, hasPointerCapture(id){return captures.has(id);}, releasePointerCapture(id){captures.delete(id);},
    };
  }
  const context = {
    document:{getElementById(id){return elements[id];}},
    ResizeObserver:class{constructor(callback){resize=callback;}observe(){}},
  };
  vm.runInNewContext(code, context);
  function fire(id, name, overrides={}) {
    const event = {button:0,pointerId:1,clientX:200,clientY:200,deltaY:0,deltaMode:0,
      target:{closest(){return null;}},preventDefault(){this.prevented=true;},...overrides};
    handlers.get(id+':'+name).handler(event);
    return event;
  }
  function transform(){
    const match = elements.image.style.transform.match(/translate\(([-\d.e+]+)px, ([-\d.e+]+)px\) scale\(([-\d.e+]+)\)/);
    assert(match, elements.image.style.transform);
    return {x:+match[1],y:+match[2],scale:+match[3]};
  }
  function covers(){
    const t=transform();
    const w=naturalWidth*t.scale, h=naturalHeight*t.scale;
    if(w < box.width) assert(Math.abs(t.x-(box.width-w)/2)<1e-6);
    else {assert(t.x<=1e-7); assert(t.x+w>=box.width-1e-7);}
    if(h < box.height) assert(Math.abs(t.y-(box.height-h)/2)<1e-6);
    else {assert(t.y<=1e-7); assert(t.y+h>=box.height-1e-7);}
  }
  covers();
  assert.strictEqual(elements.zoom.textContent,'100%');
  assert.strictEqual(elements.image.style.visibility,'visible');
  assert.strictEqual(handlers.get('viewer:wheel').options.passive,false);
  const initial=transform();
  const px=200,py=210;
  const ev=fire('viewer','wheel',{deltaY:-100,clientX:box.left+px,clientY:box.top+py});
  assert(ev.prevented);
  const zoomed=transform();
  assert(zoomed.scale>initial.scale);
  assert(Math.abs((px-initial.x)/initial.scale - (px-zoomed.x)/zoomed.scale)<1e-6);
  assert(Math.abs((py-initial.y)/initial.scale - (py-zoomed.y)/zoomed.scale)<1e-6);
  covers();
  fire('viewer','pointerdown');
  assert(elements.viewer.classList.contains('dragging'));
  fire('viewer','pointermove',{clientX:1200,clientY:-1200});
  covers();
  fire('viewer','pointermove',{clientX:-2000,clientY:2000});
  covers();
  fire('viewer','pointercancel');
  assert(!elements.viewer.classList.contains('dragging'));
  const released=transform();
  fire('viewer','pointermove',{clientX:99999});
  assert.deepStrictEqual(transform(),released);
  for(let i=0;i<20;i++)fire('viewer','wheel',{deltaY:-500});
  assert.strictEqual(elements.zoom.textContent,'600%');
  covers();
  for(let i=0;i<20;i++)fire('viewer','wheel',{deltaY:500});
  assert.strictEqual(elements.zoom.textContent,'25%');
  covers();
  fire('reset','click');
  assert.deepStrictEqual(transform(),initial);
  fire('viewer','keydown',{key:'+'});
  assert.strictEqual(elements.zoom.textContent,'120%');
  fire('reset','click');
  assert.deepStrictEqual(transform(),initial);
  fire('viewer','wheel',{deltaY:-200,deltaMode:1});
  fire('viewer','dblclick');
  assert.deepStrictEqual(transform(),initial);
  for (const height of [600,320]) {
    box={...box,width:280,height};
    resize();
    covers();
    assert.strictEqual(elements.zoom.textContent,'100%');
  }
  fire('image','error');
  assert(!elements.status.hidden);
  assert.strictEqual(elements.image.style.visibility,'hidden');
  fire('image','load');
  assert(elements.status.hidden);
  covers();
}

for(const document of documents){const code=document.match(/<script>([\s\S]*?)<\/script>/)[1];for(const [w,h] of [[300,1800],[1800,300],[600,600]])scenario(w,h,code);}
console.log('PASS: 100% initial/reset, 25%-600% zoom, centered shrink, bounded drag, keyboard, resize, errors');
