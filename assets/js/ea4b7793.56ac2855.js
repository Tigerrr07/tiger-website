"use strict";(self.webpackChunktiger_website=self.webpackChunktiger_website||[]).push([[6511],{3905:(e,t,n)=>{n.d(t,{Zo:()=>l,kt:()=>b});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=a.createContext({}),m=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},l=function(e){var t=m(e.components);return a.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},p=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,s=e.parentName,l=o(e,["components","mdxType","originalType","parentName"]),p=m(n),b=r,d=p["".concat(s,".").concat(b)]||p[b]||u[b]||i;return n?a.createElement(d,c(c({ref:t},l),{},{components:n})):a.createElement(d,c({ref:t},l))}));function b(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,c=new Array(i);c[0]=p;var o={};for(var s in t)hasOwnProperty.call(t,s)&&(o[s]=t[s]);o.originalType=e,o.mdxType="string"==typeof e?e:r,c[1]=o;for(var m=2;m<i;m++)c[m]=n[m];return a.createElement.apply(null,c)}return a.createElement.apply(null,n)}p.displayName="MDXCreateElement"},3783:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>c,default:()=>u,frontMatter:()=>i,metadata:()=>o,toc:()=>m});var a=n(7462),r=(n(7294),n(3905));const i={sidebar_position:0,title:"Static Data Member And Static Member Function"},c="Static Data Member And Static Member Function",o={unversionedId:"C++/static_data_member",id:"C++/static_data_member",title:"Static Data Member And Static Member Function",description:"Example of static data member",source:"@site/docs/C++/static_data_member.md",sourceDirName:"C++",slug:"/C++/static_data_member",permalink:"/tiger-website/docs/C++/static_data_member",draft:!1,tags:[],version:"current",sidebarPosition:0,frontMatter:{sidebar_position:0,title:"Static Data Member And Static Member Function"},sidebar:"tutorialSidebar",previous:{title:"C++ basics",permalink:"/tiger-website/docs/category/c-basics"},next:{title:"Const Data Member",permalink:"/tiger-website/docs/C++/const_data_member"}},s={},m=[{value:"Example of static data member",id:"example-of-static-data-member",level:3},{value:"Example of static member function",id:"example-of-static-member-function",level:3},{value:"From",id:"from",level:3}],l={toc:m};function u(e){let{components:t,...n}=e;return(0,r.kt)("wrapper",(0,a.Z)({},l,n,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"static-data-member-and-static-member-function"},"Static Data Member And Static Member Function"),(0,r.kt)("h3",{id:"example-of-static-data-member"},"Example of static data member"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"Static data members of a class are attributes the class."),(0,r.kt)("li",{parentName:"ul"},"The memory for static data members is common to each object.")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-cpp"},'#include <iostream>\nusing namespace std;\n\nclass Base {\npublic:\n  int x;\n  static int y; // declaration\n};\n\nint Base::y; // definition\n\nint main() {\n  Base b1, b2;\n  b1.x = 10;\n  b1.y = 30;  // Base::y\n  b2.x = 20;\n  Base::y = 40;\n\n  cout << b1.x << " " << b2.x << endl;\n  cout << b1.y << " " << b2.y << endl;\n}\n')),(0,r.kt)("p",null,"output"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"10 20\n40 40\n")),(0,r.kt)("h3",{id:"example-of-static-member-function"},"Example of static member function"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"Static member functions can only access static data members."),(0,r.kt)("li",{parentName:"ul"},"Non-static member functions can access static or non-static data memebers")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-cpp"},'#include <iostream>\nusing namespace std;\n\nclass Base {\npublic:\n  int x;\n  static int y; // declaration\n  void printXY() {cout << x << " " << y << endl;}\n  static void printY() {cout << y << endl;}\n};\n\nint Base::y; // definition\n\nint main() {\n  Base b1, b2;\n  b1.x = 10;\n  b1.y = 30;  // Base::y\n  b2.x = 20;\n  Base::y = 40;\n\n  b1.printXY();\n  b1.printY();  // Base::printY()\n\n  b2.printXY();\n  Base::printY();\n\n  return 0;\n}\n')),(0,r.kt)("p",null,"output"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"10 40\n40\n20 40\n40\n")),(0,r.kt)("h3",{id:"from"},"From"),(0,r.kt)("p",null,(0,r.kt)("a",{parentName:"p",href:"https://www.youtube.com/watch?v=u8jw0LsQFFg&list=PLk6CEY9XxSIAQ2vE_Jb4Dbmum7UfQrXgt&index=43"},"Static Data Member And Static Member Function In C++ - Youtube")))}u.isMDXComponent=!0}}]);