"use strict";(self.webpackChunktiger_website=self.webpackChunktiger_website||[]).push([[3258],{3905:(e,t,r)=>{r.d(t,{Zo:()=>m,kt:()=>s});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function l(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function o(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var d=n.createContext({}),c=function(e){var t=n.useContext(d),r=t;return e&&(r="function"==typeof e?e(t):l(l({},t),e)),r},m=function(e){var t=c(e.components);return n.createElement(d.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,i=e.originalType,d=e.parentName,m=o(e,["components","mdxType","originalType","parentName"]),u=c(r),s=a,k=u["".concat(d,".").concat(s)]||u[s]||p[s]||i;return r?n.createElement(k,l(l({ref:t},m),{},{components:r})):n.createElement(k,l({ref:t},m))}));function s(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=r.length,l=new Array(i);l[0]=u;var o={};for(var d in t)hasOwnProperty.call(t,d)&&(o[d]=t[d]);o.originalType=e,o.mdxType="string"==typeof e?e:a,l[1]=o;for(var c=2;c<i;c++)l[c]=r[c];return n.createElement.apply(null,l)}return n.createElement.apply(null,r)}u.displayName="MDXCreateElement"},3756:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>d,contentTitle:()=>l,default:()=>p,frontMatter:()=>i,metadata:()=>o,toc:()=>c});var n=r(7462),a=(r(7294),r(3905));const i={},l=void 0,o={unversionedId:"ECE408/Lecture1&2",id:"ECE408/Lecture1&2",title:"Lecture1&2",description:"Q: \u5728CUDA\u4e2d\uff0c\u4ec0\u4e48\u662fkernels?",source:"@site/docs/ECE408/Lecture1&2.md",sourceDirName:"ECE408",slug:"/ECE408/Lecture1&2",permalink:"/tiger-website/docs/ECE408/Lecture1&2",draft:!1,tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"CUDA course",permalink:"/tiger-website/docs/category/cuda-course"},next:{title:"Lecture3 Kernel-Based Data Parallel Execution Model",permalink:"/tiger-website/docs/ECE408/Lecture3"}},d={},c=[{value:"Programming Model for CUDA",id:"programming-model-for-cuda",level:2},{value:"Thread Hierarchy",id:"thread-hierarchy",level:2},{value:"gridDim",id:"griddim",level:3},{value:"blockDim",id:"blockdim",level:3},{value:"Example",id:"example",level:2}],m={toc:c};function p(e){let{components:t,...i}=e;return(0,a.kt)("wrapper",(0,n.Z)({},m,i,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,"Q: \u5728CUDA\u4e2d\uff0c\u4ec0\u4e48\u662f",(0,a.kt)("em",{parentName:"p"},"kernels"),"?"),(0,a.kt)("p",null,"A: kernels \u7c7b\u4f3c\u4e8e C++ \u4e2d functions\uff0c\u5f53\u8c03\u7528\u5b83\u65f6\uff0c\u4f1a\u5e76\u884c\u5730\u88ab\u4e0d\u540c\u7ebf\u7a0b\u6267\u884c\u3002\u5982\u679c\u6709 N \u4e2a\u7ebf\u7a0b\uff0c\u5c31\u88ab\u5e76\u884c\u5730\u6267\u884c N \u6b21\u3002"),(0,a.kt)("h2",{id:"programming-model-for-cuda"},"Programming Model for CUDA"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Each CUDA kernel",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"is executed by a ",(0,a.kt)("strong",{parentName:"li"},"grid")),(0,a.kt)("li",{parentName:"ul"},"grid is a 3D array of ",(0,a.kt)("strong",{parentName:"li"},"thread blocks"),", which are 3D arrays of ",(0,a.kt)("strong",{parentName:"li"},"threads")))),(0,a.kt)("li",{parentName:"ul"},"Each thread",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"executes the ",(0,a.kt)("strong",{parentName:"li"},"same program")," (kernel) on ",(0,a.kt)("strong",{parentName:"li"},"distinct data address")),(0,a.kt)("li",{parentName:"ul"},"has a ",(0,a.kt)("strong",{parentName:"li"},"unique identifier")," to compute memory addresses and make control decisions"),(0,a.kt)("li",{parentName:"ul"},"Single Program Multiple Data (",(0,a.kt)("strong",{parentName:"li"},"SPMD"),")")))),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"grid",src:r(1087).Z,width:"902",height:"400"})),(0,a.kt)("h2",{id:"thread-hierarchy"},"Thread Hierarchy"),(0,a.kt)("h3",{id:"griddim"},"gridDim"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"gridDim gives number of blocks",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"Number of blocks in each dimension is"),(0,a.kt)("li",{parentName:"ul"},"gridDim.x = 8, gridDim.y = 3, gridDim.z = 2"))),(0,a.kt)("li",{parentName:"ul"},"Each block has a unique index tuple",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"blockIdx.x : ","[0, griDim.x-1]"),(0,a.kt)("li",{parentName:"ul"},"blockIdx.y : ","[0, griDim.y-1]"),(0,a.kt)("li",{parentName:"ul"},"blockIdx.z : ","[0, griDim.z-1]")))),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"gridDim",src:r(2848).Z,width:"903",height:"330"})),(0,a.kt)("h3",{id:"blockdim"},"blockDim"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"blockDim gives number of threads",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"Number of blocks in each dimension is"),(0,a.kt)("li",{parentName:"ul"},"blockDim.x = 5, blockDim.y = 4, blockDim.z = 3"))),(0,a.kt)("li",{parentName:"ul"},"Each thread has a unique index tuple",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"threadIdx.x : ","[0, blockDim.x-1]"),(0,a.kt)("li",{parentName:"ul"},"threadIdx.y : ","[0, blockDim.y-1]"),(0,a.kt)("li",{parentName:"ul"},"threadIdx.z : ","[0, blockDim.z-1]")))),(0,a.kt)("admonition",{title:"NOTES",type:"tip"},(0,a.kt)("ul",{parentName:"admonition"},(0,a.kt)("li",{parentName:"ul"},"A thread block may contain up to 1024 threads."),(0,a.kt)("li",{parentName:"ul"},"\u5750\u6807\u7cfb\u662f\u56fa\u5b9a\u7684"))),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"blockDim",src:r(6755).Z,width:"1392",height:"419"})),(0,a.kt)("p",null,"\u6bcf\u4e2a threadIdx \u5728\u5b83\u7684 block \u4e2d\u662f\u552f\u4e00\u7684\uff0cthreadIdx \u52a0\u4e0a blockIdx \u5728 grid \u4e2d\u662f\u552f\u4e00\u7684\u3002"),(0,a.kt)("h2",{id:"example"},"Example"),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"threadBlocks",src:r(7645).Z,width:"1298",height:"473"}),"\n\u4e0a\u9762 vector add \u7684\u4f8b\u5b50\u4e2d block \u548c grid \u5747\u662f\u4e00\u7ef4."),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-cpp"},"__global__ void vecAdd(float* A, float* B, float* C, int n) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n)\n      C[i] = A[i] + C[i];\n}\n\nint main() {\n  // omit allocations of A, B, C, n\n  vecAdd<<<std::ceil(n/256.0), 256>>>(A, B, C, n);\n\n  // Or\n  dim3 DimGrid = (std::ceil(n/256.0), 1, 1);\n  dim3 DimBlock = (256, 1, 1);\n\n  vecAdd<<<DimGrid, DimBlock>>>(A, B, C, n);\n}\n")),(0,a.kt)("p",null,"\u6bcf\u4e2a thread \u5e76\u884c\u6267\u884c ",(0,a.kt)("inlineCode",{parentName:"p"},"vecAdd")," kernel\uff0c\u8ba1\u7b97\u6570\u636e\u7684 memory address\uff0c\u8fd9\u91cc\u6bcf\u4e2a thread \u53ea\u8ba1\u7b97\u4e00\u4e2a output element."),(0,a.kt)("admonition",{title:"NOTES",type:"tip"},(0,a.kt)("p",{parentName:"admonition"},"\u82e5\u662f\u60f3\u8ba9\u4e00\u4e2a thread \u8ba1\u7b97\u591a\u4e2a output elements\uff0c\u53ef\u4ee5\u8ba9\u4e00\u4e2a thread \u8ba1\u7b97\u591a\u4e2a memory addresses\uff0c\u4e4b\u540e\u6211\u4eec\u987a\u5e8f\u6267\u884c\u5bf9 output elements \u7684\u8ba1\u7b97\u3002")),(0,a.kt)("p",null,"\u4f8b\u5982\u6bcf\u4e2a thread \u6267\u884c\u4e24\u6b21\u52a0\u6cd5\u64cd\u4f5c\uff0c\u8fd9\u6837\u9700\u8981\u7684 thread blocks \u51cf\u5c11\u4e00\u534a\u3002\u9700\u8981\u4e24\u6b21\u83b7\u53d6\u5bf9\u5e94\u7684 ",(0,a.kt)("inlineCode",{parentName:"p"},"A")," \u548c ",(0,a.kt)("inlineCode",{parentName:"p"},"B")," \u7684 address\u3002"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-cpp"},"__global__ void vecAdd(float* A, float* B, float* C, int n) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n)\n      C[i] = A[i] + C[i];\n  i = i + blockDim.x;\n  if (i < n)\n      C[i] = A[i] + C[i];\n}\n\nint main() {\n  // omit allocations of A, B, C, n\n  vecAdd<<<std::ceil(n/2 * 256.0), 256>>>(A, B, C, n);\n}\n")))}p.isMDXComponent=!0},6755:(e,t,r)=>{r.d(t,{Z:()=>n});const n=r.p+"assets/images/blockDim-ef28a73b1a42bb1ef7106fec8e1e175b.png"},1087:(e,t,r)=>{r.d(t,{Z:()=>n});const n=r.p+"assets/images/grid-3dbc9f86692ed8f966f3913406912dd8.png"},2848:(e,t,r)=>{r.d(t,{Z:()=>n});const n=r.p+"assets/images/gridDim-420eb5bea1ac22b56007b942cc4c7d2f.png"},7645:(e,t,r)=>{r.d(t,{Z:()=>n});const n=r.p+"assets/images/oneDthreadBlocks-fa378e9343b0e735b867d834e96bd6a9.png"}}]);