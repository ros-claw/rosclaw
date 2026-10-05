import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync,mkdirSync,writeFileSync,readFileSync,rmSync,utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { continueRecentPiSession,resolveContinuationTarget,openPiSession } from "../src/harness/pi/pi-sessions.js";
import { resolveTaskContext } from "../src/native/active-task-context.js";
import { WorkspaceStore } from "../src/session/workspace.js";
function recorded(cwd:string,dir:string,provider="kimi-coding") {
 const m=SessionManager.create(cwd,dir);
 m.appendMessage({role:"user",content:"offline",timestamp:Date.now()});
 m.appendMessage({role:"assistant",content:[{type:"text",text:"offline"}],api:"anthropic-messages",provider,model:"k3",usage:{input:1,output:1,cacheRead:0,cacheWrite:0,totalTokens:2,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},stopReason:"stop",timestamp:Date.now()});
 return m;
}
function fixture(){const root=mkdtempSync(join(tmpdir(),"recorded-continue-")),repo=join(root,"repo"),home=join(root,"home"),cwd=join(repo,"nested","stage"),dir=join(home,"agent","sessions");mkdirSync(join(repo,".git"),{recursive:true});mkdirSync(cwd,{recursive:true});mkdirSync(dir,{recursive:true});return {root,repo,home,cwd,dir};}
test("nested git continuation selects recorded UUID before workspace inference",async()=>{
 const f=fixture();try {const old=recorded(f.cwd,f.dir);const store=new WorkspaceStore(f.home);store.bind(f.cwd,{normalizeToGit:false});const bytes=readFileSync(join(f.home,"agent/workspace.json"));
 const implicit=resolveTaskContext({rosclawHome:f.home,cwd:f.cwd,mode:"SIMULATION"});assert.equal(implicit.workspaceRoot,f.repo);
 assert.notEqual(SessionManager.continueRecent(implicit.workspaceRoot,f.dir).getSessionId(),old.getSessionId());
 const target=await continueRecentPiSession(f.repo,f.dir);assert.equal(target?.getSessionId(),old.getSessionId());const context=resolveTaskContext({rosclawHome:f.home,cwd:f.cwd,resumedWorkspace:target?.getCwd(),mode:"SIMULATION"});assert.equal(context.workspaceRoot,f.cwd);assert.equal(context.workspaceSource,"resumed");assert.deepEqual(readFileSync(join(f.home,"agent/workspace.json")),bytes);
 }finally{rmSync(f.root,{recursive:true,force:true});}});
test("global latest selection ignores provider and current project, preserving header identity",async()=>{const f=fixture();try {const a=recorded(f.cwd,f.dir);const b=recorded(f.repo,f.dir,"openai-codex");utimesSync(a.getSessionFile()!,new Date(1000),new Date(1000));utimesSync(b.getSessionFile()!,new Date(2000),new Date(2000));const target=await resolveContinuationTarget(f.dir);assert.equal(target?.id,b.getSessionId());assert.equal((await continueRecentPiSession(f.cwd,f.dir))?.getSessionId(),b.getSessionId());}finally{rmSync(f.root,{recursive:true,force:true});}});
test("no recorded history returns none, never allocates a fresh continuation UUID",async()=>{const f=fixture();try{assert.equal(await resolveContinuationTarget(f.dir),undefined);assert.equal(await continueRecentPiSession(f.cwd,f.dir),undefined);}finally{rmSync(f.root,{recursive:true,force:true});}});
test("same-workspace recorded history remains byte-identical on selection",async()=>{const f=fixture();try{const m=recorded(f.cwd,f.dir),before=readFileSync(m.getSessionFile()!);assert.equal((await continueRecentPiSession(f.cwd,f.dir))?.getSessionId(),m.getSessionId());assert.deepEqual(readFileSync(m.getSessionFile()!),before);}finally{rmSync(f.root,{recursive:true,force:true});}});
test("invalid or missing explicit resume never creates or rewrites workspace state",()=>{const f=fixture();try{const store=new WorkspaceStore(f.home);store.bind(f.cwd,{normalizeToGit:false});const before=readFileSync(join(f.home,"agent/workspace.json"));const corrupt=join(f.dir,"corrupt.jsonl"),empty=join(f.dir,"empty.jsonl");writeFileSync(corrupt,"invalid header\n");writeFileSync(empty,"");assert.throws(()=>openPiSession(corrupt,f.dir));assert.throws(()=>openPiSession(empty,f.dir));assert.throws(()=>openPiSession(join(f.dir,"missing.jsonl"),f.dir));assert.deepEqual(readFileSync(join(f.home,"agent/workspace.json")),before);assert.equal(readFileSync(empty,"utf8"),"");}finally{rmSync(f.root,{recursive:true,force:true});}});
test("public SDK discovery excludes corrupt headers and retains valid global target",async()=>{const f=fixture();try{const m=recorded(f.cwd,f.dir);writeFileSync(join(f.dir,"bad.jsonl"),"invalid\n");assert.equal((await resolveContinuationTarget(f.dir))?.id,m.getSessionId());}finally{rmSync(f.root,{recursive:true,force:true});}});

test("actual native failed-resume CLI exits before workspace mutation or network",()=>{const f=fixture();try{const store=new WorkspaceStore(f.home);store.bind(f.cwd,{normalizeToGit:false});const state=join(f.home,"agent/workspace.json"),before=readFileSync(state),marker=join(f.root,"network-attempt"),guard=join(f.root,"deny-network.mjs");writeFileSync(guard,`import fs from 'node:fs';globalThis.fetch=async()=>{fs.writeFileSync(${JSON.stringify(marker)},'attempt');throw Error('NETWORK_FORBIDDEN')};`);const entry=fileURLToPath(new URL("../src/main.js",import.meta.url));const child=spawnSync(process.execPath,[entry,"--resume-path",join(f.dir,"missing.jsonl")],{cwd:f.cwd,env:{...process.env,ROSCLAW_HOME:f.home,HOME:f.home,PI_OFFLINE:"1",NODE_OPTIONS:"--import="+guard},timeout:10000});assert.equal(child.error,undefined);assert.notEqual(child.status,0);assert.deepEqual(readFileSync(state),before);assert.throws(()=>readFileSync(marker));}finally{rmSync(f.root,{recursive:true,force:true});}});
