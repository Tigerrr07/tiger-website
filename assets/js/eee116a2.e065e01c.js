"use strict";(self.webpackChunktiger_website=self.webpackChunktiger_website||[]).push([[310],{3905:(e,t,n)=>{n.d(t,{Zo:()=>m,kt:()=>d});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},m=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,c=e.parentName,m=s(e,["components","mdxType","originalType","parentName"]),u=l(n),d=a,b=u["".concat(c,".").concat(d)]||u[d]||p[d]||o;return n?r.createElement(b,i(i({ref:t},m),{},{components:n})):r.createElement(b,i({ref:t},m))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=u;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:a,i[1]=s;for(var l=2;l<o;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},4252:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>i,default:()=>p,frontMatter:()=>o,metadata:()=>s,toc:()=>l});var r=n(7462),a=(n(7294),n(3905));const o={sidebar_position:1,title:"Const Data Member"},i="Const Data Member",s={unversionedId:"C++/const_data_member",id:"C++/const_data_member",title:"Const Data Member",description:"* Const data members are constant so that not to be changed once initilized.",source:"@site/docs/C++/const_data_member.md",sourceDirName:"C++",slug:"/C++/const_data_member",permalink:"/tiger-website/docs/C++/const_data_member",draft:!1,tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1,title:"Const Data Member"},sidebar:"tutorialSidebar",previous:{title:"Static Data Member And Static Member Function",permalink:"/tiger-website/docs/C++/static_data_member"},next:{title:"Polymorphsim",permalink:"/tiger-website/docs/C++/polymorphsim"}},c={},l=[{value:"Two ways of initialization",id:"two-ways-of-initialization",level:3},{value:"From",id:"from",level:3}],m={toc:l};function p(e){let{components:t,...n}=e;return(0,a.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"const-data-member"},"Const Data Member"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},"Const data members are constant so that not to be changed once initilized."),(0,a.kt)("li",{parentName:"ul"},"Each object maintains its own copy of const data members in memory, separate from other objects.")),(0,a.kt)("h3",{id:"two-ways-of-initialization"},"Two ways of initialization"),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"1st Way"),": Initialize in class"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-cpp"},"#include <iostream>\nusing namespace std;\n\nclass Circle {\npublic:\n  Circle(float a) { r = a; }\n  float getArea() { return r * r * pi; }\nprivate:\n  const float pi = 3.14;\n  float r;\n};\n\nint main() {\n  Circle c1(5.2), c2(10);\n  cout << c1.getArea() << endl;\n  cout << c2.getArea() << endl;\n  return 0;\n}\n")),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"2nd Way"),": Initializer list is used to initialize them from outside"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-cpp"},'#include <iostream>\nusing namespace std;\n\nclass Phone {\npublic:\n  Phone(string str, int a) : pname(std::move(str)), memsize(a) {};\n  string getPhoneName() {return pname; }\nprivate:\n  const string pname;\n  int memsize;\n};\n\nint main() {\n  Phone p1("HUAWEI", 64), p2("IPHONE", 32);\n  cout << p1.getPhoneName() << endl;\n  cout << p2.getPhoneName() << endl;\n  return 0;\n}\n')),(0,a.kt)("h3",{id:"from"},"From"),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://www.youtube.com/watch?v=YHr-ywZ30c0&list=PLk6CEY9XxSIAQ2vE_Jb4Dbmum7UfQrXgt&index=44"},"Const Data Member In C++ - YouTube")))}p.isMDXComponent=!0}}]);