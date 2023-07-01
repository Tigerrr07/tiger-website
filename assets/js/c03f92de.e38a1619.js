"use strict";(self.webpackChunktiger_website=self.webpackChunktiger_website||[]).push([[7421],{3905:(e,n,t)=>{t.d(n,{Zo:()=>u,kt:()=>d});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function a(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)t=i[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var p=r.createContext({}),m=function(e){var n=r.useContext(p),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},u=function(e){var n=m(e.components);return r.createElement(p.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},s=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,i=e.originalType,p=e.parentName,u=a(e,["components","mdxType","originalType","parentName"]),s=m(t),d=o,y=s["".concat(p,".").concat(d)]||s[d]||c[d]||i;return t?r.createElement(y,l(l({ref:n},u),{},{components:t})):r.createElement(y,l({ref:n},u))}));function d(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var i=t.length,l=new Array(i);l[0]=s;var a={};for(var p in n)hasOwnProperty.call(n,p)&&(a[p]=n[p]);a.originalType=e,a.mdxType="string"==typeof e?e:o,l[1]=a;for(var m=2;m<i;m++)l[m]=t[m];return r.createElement.apply(null,l)}return r.createElement.apply(null,t)}s.displayName="MDXCreateElement"},1402:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>p,contentTitle:()=>l,default:()=>c,frontMatter:()=>i,metadata:()=>a,toc:()=>m});var r=t(7462),o=(t(7294),t(3905));const i={},l="Polymorphsim",a={unversionedId:"C++/polymorphsim",id:"C++/polymorphsim",title:"Polymorphsim",description:"Polymorphsim: many forms",source:"@site/docs/C++/polymorphsim.md",sourceDirName:"C++",slug:"/C++/polymorphsim",permalink:"/tiger-website/docs/C++/polymorphsim",draft:!1,tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Const Data Member",permalink:"/tiger-website/docs/C++/const_data_member"}},p={},m=[{value:"Compile Time Polymorphsim",id:"compile-time-polymorphsim",level:2},{value:"Function overloading",id:"function-overloading",level:3},{value:"Operator overloading",id:"operator-overloading",level:3},{value:"Runtime Polymorphsim",id:"runtime-polymorphsim",level:2},{value:"From",id:"from",level:2}],u={toc:m};function c(e){let{components:n,...t}=e;return(0,o.kt)("wrapper",(0,r.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"polymorphsim"},"Polymorphsim"),(0,o.kt)("p",null,"Polymorphsim: many forms"),(0,o.kt)("p",null,"Two types of polymorphsim"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Compile Time Polymorphsim / Static Binding / Early Binding",(0,o.kt)("ul",{parentName:"li"},(0,o.kt)("li",{parentName:"ul"},"Function overloading"),(0,o.kt)("li",{parentName:"ul"},"Operator overloading"))),(0,o.kt)("li",{parentName:"ul"},"Runtime Polymorphsim / Dynamic Binding / Lazy Binding",(0,o.kt)("ul",{parentName:"li"},(0,o.kt)("li",{parentName:"ul"},"Function overriding (using virtual functions)")))),(0,o.kt)("h2",{id:"compile-time-polymorphsim"},"Compile Time Polymorphsim"),(0,o.kt)("h3",{id:"function-overloading"},"Function overloading"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-cpp"},'#include <iostream>\nusing namespace std;\n\nclass Test {\npublic:\n  void func(int x) {cout << "Integer" << endl; }\n  void func(double x) {cout << "Double" << endl; }\n};\n\nint main() {\n  Test t1;\n  t1.fun(10);\n  t1.fun(10.5);\n  return 0;\n}\n')),(0,o.kt)("p",null,"output"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"Integer\nDouble\n")),(0,o.kt)("h3",{id:"operator-overloading"},"Operator overloading"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-cpp"},'#include <iostream>\nusing namespace std;\n\nclass Complex {\npublic:\n  Complex() = default;\n  Complex(int r, int i) : real(r), imag(i) {}\n  Complex operator+(Complex const &obj) {\n    Complex res;\n    res.real = real + obj.real;\n    res.imag = imag + obj.imag;\n    return res;\n  }\n  void show() { cout << real << "+" << imag << "i" << endl; }\n\nprivate:\n  int real, imag;\n};\n\nint main() {\n  Complex c1(1, 3), c2(2, 5);\n  Complex c3 = c1 + c2;\n  c3.show();\n  return 0;\n}\n')),(0,o.kt)("p",null,"output"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"3+8i\n")),(0,o.kt)("h2",{id:"runtime-polymorphsim"},"Runtime Polymorphsim"),(0,o.kt)("p",null,"When ",(0,o.kt)("inlineCode",{parentName:"p"},"override")," keyword is used, compiler checks if the member function in the derived class matches ",(0,o.kt)("strong",{parentName:"p"},"the signature of any virtual function")," in the base class."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-cpp"},'#include <iostream>\nusing namespace std;\n\nclass Base {\npublic:\n  virtual void fun() { cout << "Base" << endl; }\n};\n\nclass Derived : public Base {\npublic:\n  void fun() override { cout << "Derived" << endl; }\n};\n\nint main() {\n  Base *a = new Derived();\n  a->fun();\n\n  Derived b;\n  Base &c = b;\n  c.fun();\n  return 0;\n}\n')),(0,o.kt)("p",null,"output"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"Derived\nDerived\n")),(0,o.kt)("h2",{id:"from"},"From"),(0,o.kt)("p",null,(0,o.kt)("a",{parentName:"p",href:"https://www.youtube.com/watch?v=mv5_l4kuVho&list=PLk6CEY9XxSIAQ2vE_Jb4Dbmum7UfQrXgt&index=72"},"Polymorphism In C++ | Static ","&"," Dynamic Binding | Lazy ","&"," Early Binding In C++ - YouTube")))}c.isMDXComponent=!0}}]);