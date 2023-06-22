"use strict";(self.webpackChunktiger_website=self.webpackChunktiger_website||[]).push([[6589],{8733:e=>{e.exports=JSON.parse('{"blogPosts":[{"id":"/2023/6/20/VAE/VAE","metadata":{"permalink":"/tiger-website/blog/2023/6/20/VAE/VAE","source":"@site/blog/2023-6-20-VAE/VAE.md","title":"VAE","description":"Unconditional model","date":"2023-06-20T00:00:00.000Z","formattedDate":"June 20, 2023","tags":[],"readingTime":6.045,"hasTruncateMarker":false,"authors":[{"name":"Hu Chen","title":"Master Candidate @ SDU","url":"https://github.com/Tigerrr07","imageURL":"https://github.com/Tigerrr07.png","key":"tiger"}],"frontMatter":{"title":"VAE","authors":"tiger"},"nextItem":{"title":"\u5f52\u4e00\u5316","permalink":"/tiger-website/blog/2023/6/17/Normalization"}},"content":"## Unconditional model\\r\\n\u5047\u8bbe\u6709\u89c2\u6d4b\u53d8\u91cf $\\\\bold{x}$, \u771f\u5b9e\u5206\u5e03 $p(\\\\bold{x})$ \u662f\u672a\u77e5\u7684\uff0c\u6211\u4eec\u60f3\u7528\u6a21\u578b $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x})$ \u53bb\u8fd1\u4f3c\u8fd9\u4e2a\u672a\u77e5\u5206\u5e03\uff0c\u53c2\u6570\u4e3a $\\\\boldsymbol{\\\\theta}$ :\\r\\n\\r\\n$$\\r\\np_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}) \\\\approx p(\\\\bold{x}) \\\\tag{1}\\r\\n$$\\r\\n\\r\\n> TODO: \u6700\u5927\u5316\u6781\u5927\u4f3c\u7136 \u6700\u5927\u5316\u5bf9\u6570\u4f3c\u7136\\r\\n> ML = \u6700\u5c0f\u5316 KL\\r\\n> ML and MAP\\r\\n\\r\\n\u6211\u4eec\u53ef\u4ee5\u4f7f\u7528\u6700\u5927\u4f3c\u7136 (Maximum Likelihood) \u53bb\u627e\u5230\u53c2\u6570\u4e3a $\\\\boldsymbol{\\\\theta}$ \uff0c\u4ece\u800c\u5f97\u5230\u6211\u4eec\u7684\u6a21\u578b $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x})$\u3002\\r\\n\\r\\n\u4f46\u662f\u6211\u4eec\u4e0d\u77e5\u9053 $p(\\\\bold{x})$ \u662f\u4ec0\u4e48\u6837\u7684\uff0c\u4f8b\u5982\u5982\u679c\u5176\u5206\u5e03\u7b26\u5408\u591a\u5143\u9ad8\u65af\u5206\u5e03\uff0c\u5219\u53ef\u4ee5\u6bd4\u8f83\u5bb9\u6613\u7684\u6c42\u51fa\u3002\u4f46\u771f\u5b9e\u7684  $p(\\\\bold{x})$ \u5f80\u5f80\u662f\u975e\u5e38\u590d\u6742\u7684\uff0c$\\\\bold{x}$ \u53ef\u4ee5\u662f\u6587\u672c\uff0c\u56fe\u7247\uff0c\u751a\u81f3\u662fgraph\u3002\\r\\n\\r\\n\u6211\u4eec\u77e5\u9053\u795e\u7ecf\u7f51\u7edc\u6a21\u578b (Neural Network, NN) \u80fd\u591f\u8868\u793a\u590d\u6742\u7684\u6a21\u578b\uff0c\u5982\u679c\u4f7f\u7528NN \u8868\u793a $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x})$\uff0cNN \u7684\u53c2\u6570\u4e3a $\\\\boldsymbol{\\\\theta}$\u3002\u4e0b\u9762\u4ee5\u4e00\u4e2a\u6837\u672c\u4e3a\u4f8b\uff0c\\r\\n* \u5047\u8bbe $\\\\bold{x}_i \\\\sim p(\\\\bold{x})$ \u662f\u4e00\u4e2a\u6837\u672c\uff0c\u6211\u4eec\u60f3\u8981\u6700\u5927\u4f3c\u7136 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}_i)$\uff0c\u5728 NN \u4e2d\u6211\u4eec\u4f7f\u7528\u68af\u5ea6\u4e0b\u964d\u4f18\u5316\u76ee\u6807\u51fd\u6570\\r\\n* \u6211\u4eec\u60f3\u8981\u6c42 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}_i)$ \u7684\u68af\u5ea6\uff0c\u53bb\u66f4\u65b0 NN \u7684\u53c2\u6570\uff0c\u4ece\u800c\u8fbe\u5230\u6700\u5927\u4f3c\u7136\u7684\u76ee\u7684\\r\\n* \u5c06\u5df2\u77e5\u6837\u672c $\\\\bold{x}_i$ \u4f5c\u4e3a NN \u8f93\u5165\uff0c$\\\\boldsymbol{\\\\theta}$ \u4f5c\u4e3a NN \u53c2\u6570\uff0c\u8f93\u51fa\u4e3a\u4e00\u4e2a\u9884\u6d4b\u7684\u6982\u7387\u503c\uff0c\u4f46\u662f\u6211\u4eec\u6ca1\u6709\u76d1\u7763\u4fe1\u606f\uff0c\u65e0\u6cd5\u8bbe\u8ba1\u635f\u5931\u51fd\u6570\\r\\n\\r\\n\u56e0\u6b64 $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ is intractable.\\r\\n\\r\\n## Deep Latent Variable Models\\r\\nFor the intractbility of   $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$\uff0c\u6211\u4eec\u5f15\u5165\u9690\u53d8\u91cf $\\\\bold{z}$\uff0c\u5bf9\u4e8e\u65e0\u6761\u4ef6\u7684\u89c2\u6d4b\u53d8\u91cf $\\\\bold{x}$ \uff0c\u52a0\u5165\u9690\u53d8\u91cf $\\\\bold{z}$\uff0c\u5f97\u5230\u8054\u5408\u5206\u5e03 $p_{\\\\bold{\\\\theta}}(\\\\bold{x}, \\\\bold{z})$\uff0c\u79f0\u4e3a\u9690\u53d8\u91cf\u6a21\u578b\uff0c\u6b64\u65f6 $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ \u53ef\u7528\u8fb9\u9645\u5206\u5e03\u6765\u8868\u793a\uff1a\\r\\n\\r\\n$$\\r\\np_{\\\\bold{\\\\theta}}(\\\\bold{x}) = \\\\int p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z}) d\\\\bold{z} \\\\tag{2}\\r\\n$$\\r\\n\\r\\n\u4e5f\u79f0\u4e3a marginal likelihood \u6216 model evidence\u3002\u8fd9\u79cd\u5728 $\\\\bold{x}$ \u4e0a\u9690\u5f0f\u7684\u5206\u5e03\u975e\u5e38\u7075\u6d3b\u3002\\r\\n\\r\\n\u6211\u4eec\u77e5\u9053\u5bf9\u4e8e\u9690\u53d8\u91cf\u6a21\u578b\uff1a\\r\\n\\r\\n$$\\r\\np_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z}) = p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z}) p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z}) \\\\tag{3}\\r\\n$$\\r\\n\\r\\n$p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z})$ \u79f0\u4e3a\u5148\u9a8c\u5206\u5e03 (prior distribution)\uff0c$p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})$ \u662f\u6761\u4ef6\u5206\u5e03 (conditional distribution)\u3002\\r\\n\\r\\n* \u82e5 $\\\\bold{z}$ \u662f\u79bb\u6563\u7684\uff0c\u4e14 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})$ \u662f\u9ad8\u65af\u5206\u5e03\uff0c\u5219 $\\\\bold{x}$ \u662f\u6709\u9650\u6df7\u5408\u9ad8\u65af\u5206\u5e03\\r\\n*  \u82e5 $\\\\bold{z}$ \u662f\u8fde\u7eed\u7684\uff0c\u4e14 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})$ \u662f\u9ad8\u65af\u5206\u5e03\uff0c\u5219 $\\\\bold{x}$ \u662f\u65e0\u9650\u6df7\u5408\u9ad8\u65af\u5206\u5e03\\r\\n\\r\\n\u82e5\u7528 NN \u8868\u793a $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z})$\uff0c$\\\\boldsymbol{\\\\theta}$ \u662f NN \u7684\u53c2\u6570\uff0c\u6211\u4eec\u79f0\u5176\u4e3a *deep latent variable model* (DLVM)\u3002\u5373\u5148\u9a8c\u6216\u8005\u6761\u4ef6\u5206\u5e03\u8db3\u591f\u7b80\u5355\uff0c\u4f8b\u5982\u5c06 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z})$ \u8868\u793a\u4e3a\u9ad8\u65af\u5206\u5e03\uff0c$p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})$ \u8868\u793a\u4e3a\u4f2f\u52aa\u5229\u5206\u5e03\u3002\u5219\u5f97\u5230\u7684\u8fb9\u9645\u5206\u5e03 $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ \u4ecd\u7136\u8db3\u591f\u590d\u6742\uff0c\u6a21\u578b\u5177\u6709\u5f88\u5f3a\u7684\u8868\u8fbe\u80fd\u529b\u3002\\r\\n\\r\\n## Example DLVM for multivariate Bernoulli data\\r\\n\u5047\u8bbe $D$ \u7ef4\u4e8c\u5143\u6570\u636e $\\\\bold{x} \\\\in \\\\{0,1\\\\}^D$\uff0c\u6211\u4eec\u8ba9\u5148\u9a8c\u7684 PDF \u4e3a $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z}) = N(\\\\bold{z};0,\\\\bold{I})$\uff0c\u6211\u4eec\u53ef\u4ee5\u628a $\\\\boldsymbol{\\\\theta}$ \u53bb\u6389\uff0c\u56e0\u4e3a\u6ca1\u6709\u53c2\u6570\u3002\u8ba9 $\\\\text{log}p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})$ \u662f\u4e00\u4e2a\u591a\u5143\u4f2f\u52aa\u5229\u5206\u5e03\uff0c\u6bcf\u4e00\u7ef4\u72ec\u7acb\uff0c\u5176\u6982\u7387\u4f7f\u7528 NN \u4ece $\\\\bold{z}$ \u8ba1\u7b97\u51fa\uff0c\\r\\n$$\\r\\np(\\\\bold{z}) = N(\\\\bold{z};0,\\\\bold{I}) \\\\tag{4}\\r\\n$$\\r\\n\\r\\n$$\\r\\n\\\\bold{p} = \\\\text{Decoder}_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z})  \\\\tag{5}\\r\\n$$\\r\\n\u5176\u4e2d Decoder \u7684\u6700\u540e\u4e00\u5c42\u63a5\u4e86\u4e00\u4e2a sigmoid \u51fd\u6570\uff0c$\\\\forall p_j \\\\in  \\\\bold{p}: 0 \\\\leq p_j \\\\leq 1$\uff0c$\\\\bold{x}$ \u662f\u4e00\u4e2a\u6837\u672c\uff0c\u6211\u4eec\u8981\u6700\u5927\u5982\u4e0b\u7684\u5bf9\u6570\u4f3c\u7136\uff1a\\r\\n$$\\r\\n\\\\text{log}p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z}) = \\\\sum_{j=1}^D \\\\text{log}p_{\\\\boldsymbol{\\\\theta}}(x_j | \\\\bold{z})  \\\\tag{6}\\r\\n$$\\r\\n\\r\\n$p_{\\\\boldsymbol{\\\\theta}}(x_j | \\\\bold{z})$ \u662f\u4e00\u4e2a\u4e8c\u5143\u7684\u4f2f\u52aa\u5229\u5206\u5e03\uff0c\\r\\n$$\\r\\np_{\\\\boldsymbol{\\\\theta}}(x_j | \\\\bold{z}) = \\r\\n\\\\begin{cases}\\r\\np_j & \\\\text{if }x_j = 1 \\\\\\\\\\r\\n1 - p_j & \\\\text{if }x_j = 0\\r\\n\\\\end{cases}\\r\\n\\\\tag{7}\\r\\n$$\\r\\n\u4f7f\u7528\u7edf\u4e00\u5f62\u5f0f\u53ef\u4ee5\u8868\u793a\u4e3a\uff0c\\r\\n$$\\r\\np_{\\\\boldsymbol{\\\\theta}}(x_j | \\\\bold{z}) = p_j^{x_j} (1 - p_j)^{(1-x_j)} \\\\tag{8}\\r\\n$$\\r\\n\\r\\n\\r\\n\\r\\n\u5e26\u5165 $(6)$ \u4e2d\u53ef\u4ee5\u5f97\u5230\uff0c\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\text{log}p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z}) &= \\\\sum_{j=1}^D \\\\text{log}p_{\\\\boldsymbol{\\\\theta}}(x_j | \\\\bold{z}) \\\\\\\\\\r\\n&= \\\\sum_{j=1}^D x_j \\\\log p_j + (1-x_j) \\\\log (1-p_j)\\r\\n\\\\end{aligned}\\r\\n\\\\tag{9}\\r\\n$$\\r\\n\\r\\n\u56e0\u6b64\u5bf9\u4e8e $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z}) = p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z}) p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})=p(\\\\bold{z}) p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})$\\r\\n\u662f\u53ef\u4ee5\u6c42\u68af\u5ea6\u7684\uff0c\u6211\u4eec\u6700\u5927\u5176\u5bf9\u6570\u4f3c\u7136\uff0c\\r\\n\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\nabla_{\\\\boldsymbol{\\\\theta}} \\\\log p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z}) &= \\\\nabla_{\\\\boldsymbol{\\\\theta}} \\\\log (p(\\\\bold{z}) p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})) \\\\\\\\\\r\\n&= \\\\nabla_{\\\\boldsymbol{\\\\theta}} (\\\\log p(\\\\bold{z}) + \\\\log p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})) \\\\\\\\\\r\\n&= \\\\nabla_{\\\\boldsymbol{\\\\theta}} \\\\log p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} | \\\\bold{z})\\r\\n\\\\end{aligned}\\r\\n\\\\tag{10}\\r\\n$$\\r\\n\\r\\n\u56e0\u6b64 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z})$ is tractable\u3002\\r\\n\\r\\n\u6839\u636e\u4e0a\u8ff0\u5047\u5b9a\uff0c\\r\\n\\r\\n$$\\r\\n\\\\nabla_{\\\\boldsymbol{\\\\theta}} \\r\\np_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}) = \\\\int \\\\nabla_{\\\\boldsymbol{\\\\theta}} p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} , \\\\bold{z}) d\\\\bold{z} \\\\tag{11}\\r\\n$$\\r\\n\\r\\n\\r\\n\\r\\n\u4e4b\u524d\u6211\u4eec\u77e5\u9053 $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ intractable\uff0c\u4f46\u662f\u5982\u679c\u6211\u4eec\u5f15\u5165 DLVM\uff0c\u53ea\u8981\u80fd\u591f\u5bf9 $(11)$ \u79ef\u5206\uff0c\u6211\u4eec\u5c31\u53ef\u4ee5\u5f97\u5230 $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ \u7684\u68af\u5ea6\uff0c\u4ece\u800c\u4f7f\u5f97 $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ tractable\u3002\\r\\n\\r\\n## Intractabilities\\r\\n\\r\\n\u4f46 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x}, \\\\bold{z})$ \u662f\u4e00\u4e2a NN \u6a21\u578b\uff0c\u65e0\u6cd5\u5bf9\u5176\u6c42\u79ef\u5206\uff0c\u56e0\u6b64 $(11)$ \u4e2d\u7684\u79ef\u5206\u6ca1\u6709\u89e3\u6790\u89e3\uff0c\u6211\u4eec\u5c31\u65e0\u6cd5\u8ba1\u7b97\u68af\u5ea6\u3002\\r\\n\u6b64\u5916\uff0cthe intractability of $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$ \u4e0e \u540e\u9a8c\u5206\u5e03 (Posterior Distributiion) $p_{\\\\bold{\\\\theta}}(\\\\bold{z}|\\\\bold{x})$  \u7684 intractability \u6709\u5173\uff0c\\r\\n\\r\\n$$\\r\\np_{\\\\bold{\\\\theta}}(\\\\bold{z}|\\\\bold{x}) = \\\\frac{p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} , \\\\bold{z})}{p_{\\\\bold{\\\\theta}}(\\\\bold{x})} \\\\tag{12}\\r\\n$$\\r\\n\u8054\u5408\u5206\u5e03 $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x} , \\\\bold{z})$ \u7531\u4e4b\u524d\u7684\u4f8b\u5b50\u6211\u4eec\u5df2\u7ecf\u53ef\u4ee5\u7b97\u51fa\uff0ctractable\u7684posterior $p_{\\\\bold{\\\\theta}}(\\\\bold{z}|\\\\bold{x})$ \u4f1a\u5bfc\u81f4 tractable \u7684 marginal likelihood $p_{\\\\bold{\\\\theta}}(\\\\bold{x})$\uff0c\u53cd\u4e4b\u4ea6\u7136\u3002\\r\\n\\r\\n\u4e3a\u4e86\u5c06 DLVM intractable\u7684\u540e\u9a8c\u548c\u5b66\u4e60\u95ee\u9898\u8f6c\u5316\u4e3atractable\u95ee\u9898\uff0c\u4f7f\u7528\u8fd1\u4f3c\u63a8\u65ad\u7684\u6280\u672f\u3002\\r\\n\\r\\n## Encoder or Approximate Posterior\\r\\n\u6211\u4eec\u5f15\u5165\u4e00\u4e2a\u53c2\u6570\u63a8\u65ad\u6a21\u578b (*inference model*) $q_{\\\\boldsymbol{\\\\phi}}(\\\\bold{z}|\\\\bold{x})$ \u53bb\u8fd1\u4f3c\u540e\u9a8c\uff0c\u8fd9\u4e2a\u6a21\u578b\u4e5f\u79f0\u4e3a *encoder*\uff0c$\\\\boldsymbol{\\\\phi}$ \u662f\u63a8\u65ad\u6a21\u578b\u7684\u53c2\u6570\uff0c\u79f0\u4e3a \u53d8\u5206\u53c2\u6570 (*variational parameters*)\uff0c\\r\\n$$\\r\\nq_{\\\\boldsymbol{\\\\phi}}(\\\\bold{z}|\\\\bold{x}) \\\\approx p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{z}|\\\\bold{x}) \\\\tag{13}\\r\\n$$\\r\\n\\r\\n\u8fd9\u79cd\u5bf9\u540e\u9a8c\u7684\u8fd1\u4f3c\u4f1a\u5e2e\u52a9\u6700\u5927\u5316 marginal likelihood $p_{\\\\boldsymbol{\\\\theta}}(\\\\bold{x})$\u3002\\r\\n\\r\\n\u7c7b\u4f3c\u4e8eDLVM\uff0c\u53ef\u4ee5\u4f7f\u7528 NN \u8868\u793a $q_{\\\\boldsymbol{\\\\phi}}(\\\\bold{z}|\\\\bold{x})$\uff0c$\\\\boldsymbol{\\\\phi}$ \u662f NN \u7684\u53c2\u6570\uff0c\u4f8b\u5982\\r\\n\\r\\n$$\\r\\n(\\\\boldsymbol{\\\\mu},\\\\log \\\\boldsymbol{\\\\sigma}) = \\\\text{Encoder}_{\\\\boldsymbol{\\\\phi}}(\\\\bold{x}) \\\\tag{14}\\r\\n$$\\r\\n\\r\\n$$\\r\\nq_{\\\\boldsymbol{\\\\phi}}(\\\\bold{z}|\\\\bold{x}) = N(\\\\bold{z};\\\\boldsymbol{\\\\mu}, \\\\text{diag} (\\\\boldsymbol{\\\\sigma}) )\\\\tag{15}\\r\\n$$\\r\\n> $\\\\boldsymbol{\\\\sigma}$ \u4e3a\u6807\u51c6\u5dee $\\\\geq 0$\uff0c\u8ba9 NN \u8f93\u51fa\u4e3a\u603b\u662f $\\\\geq 0$ \u662f\u56f0\u96be\u7684\uff0c\u56e0\u6b64\u52a0\u5165 $\\\\log \\\\boldsymbol{\\\\sigma}$ \u4f7f\u5f97 NN \u8f93\u51fa\u4e0d\u53d7\u9650\u5236\u3002"},{"id":"/2023/6/17/Normalization","metadata":{"permalink":"/tiger-website/blog/2023/6/17/Normalization","source":"@site/blog/2023-6-17-Normalization.md","title":"\u5f52\u4e00\u5316","description":"\u63d0\u9ad8\u6536\u655b\u901f\u5ea6\u3002","date":"2023-06-17T00:00:00.000Z","formattedDate":"June 17, 2023","tags":[{"label":"Normalization","permalink":"/tiger-website/blog/tags/normalization"}],"readingTime":1.14,"hasTruncateMarker":false,"authors":[{"name":"Hu Chen","title":"Master Candidate @ SDU","url":"https://github.com/Tigerrr07","imageURL":"https://github.com/Tigerrr07.png","key":"tiger"}],"frontMatter":{"title":"\u5f52\u4e00\u5316","authors":"tiger","tags":["Normalization"]},"prevItem":{"title":"VAE","permalink":"/tiger-website/blog/2023/6/20/VAE/VAE"},"nextItem":{"title":"zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","permalink":"/tiger-website/blog/zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528"}},"content":"\u63d0\u9ad8\u6536\u655b\u901f\u5ea6\u3002\\r\\n\\r\\n\\r\\n## Batch Normalization\\r\\n\u6279\u91cf\u5f52\u4e00\u5316 (Batch Normalization)\uff0c\u5bf9\u4e8e\u4e00\u4e2aDNN\uff0c\u7b2c $l$ \u5c42\u7684\u51c0\u8f93\u5165\u4e3a $\\\\boldsymbol{z}^{(l)}$ \u7ecf\u8fc7\u4eff\u5c04\u53d8\u6362 (Affine Transformation) $=\\\\boldsymbol{W}\\\\boldsymbol{a}^{(l-1)}+\\\\boldsymbol{b}$\uff0c\u6fc0\u6d3b\u51fd\u6570 $f(\xb7)$\uff0c\\r\\n$$\\r\\n\\\\boldsymbol{a}^{(l)}= f(\\\\boldsymbol{W}\\\\boldsymbol{a}^{(l-1)}+\\\\boldsymbol{b})\\r\\n$$\\r\\n\\r\\nIn practice\uff0cBN before Affine Transformation, after activation function.\\r\\n\u5bf9\u4e00\u4e2a\u4e2d\u95f4\u5c42\u7684\u5355\u4e2a\u795e\u7ecf\u5143\u8fdb\u884c\u5f52\u4e00\u5316\uff0c\u4f7f\u7528 Standardization \u5c06\u51c0\u8f93\u5165 $\\\\boldsymbol{z}^{l}$\\r\\n\\r\\n## Layer Normalization\\r\\n\u5c42\u5f52\u4e00\u5316 (Layer Normalization)\\r\\n\u5bf9\u4e00\u4e2a\u4e2d\u95f4\u5c42\u7684\u6240\u6709\u795e\u7ecf\u5143\u8fdb\u884c\u5f52\u4e00\u5316\uff0c\\r\\n\u5728RNN\u4e2d\uff0c\u51c0\u8f93\u5165 $\\\\boldsymbol{z}_t$ \u7531\u4e8e\u4f1a\u7d2f\u52a0\u524d\u4e00\u65f6\u523b\u7684\u72b6\u6001\uff0c\u4f1a\u968f\u7740\u65f6\u95f4\u6162\u6162\u53d8\u5927\u6216\u53d8\u5c0f\uff0c\u4ece\u800c\u5bfc\u81f4\u68af\u5ea6\u7206\u70b8\u6216\u8005\u6d88\u5931\uff0cLN\u53ef\u4ee5\u7f13\u89e3\u3002\\r\\n\\r\\n$K$ \u4e2a\u6837\u672c\u7684 mini-batch $\\\\boldsymbol{Z}^{(l)}=[\\\\boldsymbol{z}^{(1,l)};...;\\\\boldsymbol{z}^{(K,l)}]$\uff0c\u5176\u4e2d\u6bcf\u4e2a\u6837\u672c\u7684\u7279\u5f81\u5411\u91cf\u7528\u5217\u5411\u91cf\u8868\u793a\uff0cBN \u662f\u5bf9\u77e9\u9635\u7684 $\\\\boldsymbol{Z}^{(l)}$ \u7684\u6bcf\u4e00\u884c\u8fdb\u884c\u5f52\u4e00\u5316\uff0cLN\u662f\u5bf9\u77e9\u9635\u7684\u6bcf\u4e00\u5217\u8fdb\u884c\u5f52\u4e00\u5316\u3002\\r\\n## Weight Normalization\\r\\n\u4e0d\u5bf9\u51c0\u8f93\u5165\u8fdb\u884c\u5f52\u4e00\u5316\uff0c\u5bf9\u6743\u91cd\u8fdb\u884c\u5f52\u4e00\u5316\u3002"},{"id":"zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","metadata":{"permalink":"/tiger-website/blog/zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","source":"@site/blog/2023-2-19-zsh/zsh&oh-my-zsh.md","title":"zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","description":"zsh\u662f\u529f\u80fd\u66f4\u5f3a\u5927\u7684\u547d\u4ee4\u89e3\u91ca\u5668\uff0clinux\u9ed8\u8ba4\u7684\u547d\u4ee4\u89e3\u91ca\u5668\u662fbash\u3002","date":"2023-02-19T00:00:00.000Z","formattedDate":"February 19, 2023","tags":[{"label":"zsh","permalink":"/tiger-website/blog/tags/zsh"}],"readingTime":1.33,"hasTruncateMarker":false,"authors":[{"name":"Hu Chen","title":"Master Candidate @ SDU","url":"https://github.com/Tigerrr07","imageURL":"https://github.com/Tigerrr07.png","key":"tiger"}],"frontMatter":{"slug":"zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","title":"zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","authors":"tiger","tags":["zsh"]},"prevItem":{"title":"\u5f52\u4e00\u5316","permalink":"/tiger-website/blog/2023/6/17/Normalization"},"nextItem":{"title":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","permalink":"/tiger-website/blog/\u62c9\u666e\u62c9\u65af\u77e9\u9635"}},"content":"zsh\u662f\u529f\u80fd\u66f4\u5f3a\u5927\u7684\u547d\u4ee4\u89e3\u91ca\u5668\uff0clinux\u9ed8\u8ba4\u7684\u547d\u4ee4\u89e3\u91ca\u5668\u662fbash\u3002\\r\\n\\r\\n## zsh\u4e0b\u8f7d\\r\\n\\r\\n\u67e5\u770b\u7cfb\u7edf\u4e2d\u5b89\u88c5\u7684shell\u6709\u54ea\u4e9b\uff1a\\r\\n\\r\\n```bash\\r\\ncat /etc/shells\\r\\n```\\r\\n\\r\\n\u82e5\u6ca1\u6709\u5219\u4e0b\u8f7dzsh:\\r\\n\\r\\n```bash\\r\\nsudo apt install zsh\\r\\n```\\r\\n\\r\\n## oh-my-zsh\\r\\n\\r\\noh-my-zsh\u662f\u4e00\u4e2a\u5df2\u7ecf\u914d\u7f6e\u6587\u4ef6\uff0c\u5e2e\u52a9\u6211\u4eec\u914d\u7f6ezsh.\\r\\n### \u4e0b\u8f7d\\r\\n\\r\\n1. \u628a oh-my-zsh \u9879\u76eeclone\u5230\u7528\u6237\u76ee\u5f55\\r\\n\\r\\n    ```bash\\r\\n    git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh\\r\\n    ```\\r\\n\\r\\n2. \u590d\u5236\u6a21\u677f\u5230\u7528\u6237\u76ee\u5f55\u4e0b\u7684.zshrc\u6587\u4ef6\\r\\n\\r\\n    ```bash\\r\\n    cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc\\r\\n    ```\\r\\n\\r\\n3. \u66f4\u6539\u9ed8\u8ba4\u7684shell\\r\\n\\r\\n    ```bash\\r\\n    chsh -s /bin/zsh\\r\\n    ```\\r\\n\\r\\n\u4e4b\u540e.zshrc\u5c31\u66ff\u6362\u6389\u4e86\u539f\u6765\u7684.bashrc\\r\\n\\r\\n### \u4e3b\u9898\u914d\u7f6e\\r\\n\\r\\n\u4f7f\u7528VSCode\u6216\u8005Vim\u7f16\u8f91~./zshrc\u6587\u4ef6\uff0c\u66f4\u6539\u4e3b\u9898\u53ea\u9700\u66ff\u6362`ZSH_THEME`\u200b\u200b\\r\\n![Theme](./fig1.png)\\r\\n\\r\\n\\r\\n### conda\u547d\u4ee4\u8865\u5168\\r\\n\\r\\n1. \u4e0b\u8f7d\u5bf9\u5e94\u63d2\u4ef6\u5230`.oh-my-zsh`\u200b\u6587\u4ef6\u5939\u4e0b\uff1a\\r\\n\\r\\n    ```bash\\r\\n    git clone https://github.com/esc/conda-zsh-completion $ZSH_CUSTOM/plugins/conda-zsh-completion\\r\\n    ```\\r\\n\\r\\n2. \u4fee\u6539`.zshrc`\u200b\u6587\u4ef6\\r\\n\\r\\n    \u5728\u521d\u59cb\u5316 `oh-my-zsh`\u200b \u547d\u4ee4\u524d\u52a0\u5165\\r\\n\\r\\n    ```bash\\r\\n    fpath+=$ZSH_CUSTOM/plugins/conda-zsh-completion\\r\\n    ```\\r\\n    ![.zshrc1](./fig2.png)\\r\\n\\r\\n\\r\\n\u6700\u540e\u5728\u6587\u4ef6\u672b\u5c3e\u4e2d\u52a0\u5165\\r\\n\\r\\n```bash\\r\\ncompinit conda\\r\\n```\\r\\n![.zshrc2](./fig3.png)\\r\\n\u200d\\r\\nconda\u547d\u4ee4\u73b0\u5728\u611f\u89c9\u6709bug... \u4e0d\u662f\u5f88\u597d\u7528\uff0c\u6211\u53c8\u53d6\u6d88\u4e86\uff0c\u8fd8\u662f\u624b\u52a8\u6253\u5427\u3002\\r\\n\\r\\n\u53c2\u8003\\r\\n[zsh &amp; oh-my-zsh \u7684\u914d\u7f6e\u4e0e\u4f7f\u7528 - \u77e5\u4e4e (zhihu.com)](https://zhuanlan.zhihu.com/p/58073103)"},{"id":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","metadata":{"permalink":"/tiger-website/blog/\u62c9\u666e\u62c9\u65af\u77e9\u9635","source":"@site/blog/2022-11-2-Laplacian.md","title":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","description":"\u9ed8\u8ba4\u5411\u91cf\u90fd\u662f\u5217\u5411\u91cf\u3002\u6c42\u548c\u7b26\u53f7\u53ef\u4ee5\u7b80\u5199\u3002","date":"2022-11-02T00:00:00.000Z","formattedDate":"November 2, 2022","tags":[{"label":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","permalink":"/tiger-website/blog/tags/\u62c9\u666e\u62c9\u65af\u77e9\u9635"},{"label":"\u62c9\u666e\u62c9\u65af\u7279\u5f81\u6620\u5c04","permalink":"/tiger-website/blog/tags/\u62c9\u666e\u62c9\u65af\u7279\u5f81\u6620\u5c04"}],"readingTime":4.075,"hasTruncateMarker":false,"authors":[{"name":"Hu Chen","title":"Master Candidate @ SDU","url":"https://github.com/Tigerrr07","imageURL":"https://github.com/Tigerrr07.png","key":"tiger"}],"frontMatter":{"slug":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","title":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","authors":"tiger","tags":["\u62c9\u666e\u62c9\u65af\u77e9\u9635","\u62c9\u666e\u62c9\u65af\u7279\u5f81\u6620\u5c04"]},"prevItem":{"title":"zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528","permalink":"/tiger-website/blog/zsh & oh-my-zsh \u914d\u7f6e\u548c\u4f7f\u7528"},"nextItem":{"title":"Welcome","permalink":"/tiger-website/blog/welcome"}},"content":"> \u9ed8\u8ba4\u5411\u91cf\u90fd\u662f\u5217\u5411\u91cf\u3002\u6c42\u548c\u7b26\u53f7\u53ef\u4ee5\u7b80\u5199\u3002\\r\\n\\r\\n## \u62c9\u666e\u62c9\u65af\u77e9\u9635\\r\\n\\r\\n### \u5b9a\u4e49\\r\\n\\r\\n\u65e0\u5411\u56fe$G=(V,E)$\uff0c$A \\\\in \\\\mathbb{R}^{n \\\\times n}$ \u4e3a\u90bb\u63a5\u77e9\u9635\uff0c\u5176\u5143\u7d20\\r\\n$$\\r\\na_{ij}=\\\\begin{cases}\\r\\n1 & \\\\mathrm{if}\\\\ (v_i,v_j) \\\\in E \\\\\\\\\\r\\n0 & \\\\mathrm{else}\\r\\n\\\\end{cases}\\r\\n$$\\r\\n\\r\\n$N(i)$\u4e3a\u7ed3\u70b9$v_i$\u7684\u90bb\u5c45\uff0c$D \\\\in \\\\mathbb{R}^{n \\\\times n}$\u4e3a\u5ea6\u77e9\u9635\uff0c\u5bf9\u89d2\u77e9\u9635\uff0c\u5176\u5143\u7d20\\r\\n$$\\r\\nd_{ii}= \\\\sum_{j=1}^n a_{ij}= \\\\sum _{j \\\\in N(i)} a_{ij}\\r\\n$$\\r\\n\\r\\n\u5b9a\u4e49\u62c9\u666e\u62c9\u65af\u77e9\u9635 (Laplacian matrix) $L=D-A$\uff0c\u5176\u5143\u7d20\\r\\n$$\\r\\nl_{ij}=\\r\\n\\\\begin{cases}\\r\\nd_i & \\\\mathrm{if}\\\\ i=j \\\\\\\\\\r\\n-1 & \\\\mathrm{if}\\\\ (v_i,v_j) \\\\in E  \\\\\\\\\\r\\n0 & \\\\mathrm{otherwise}\\r\\n\\\\end{cases}\\r\\n$$\\r\\n\\r\\n\u6b63\u5219\u5316\u8868\u8fbe\u5f62\u5f0f (symmetric normalized laplacian) $L_{\\\\mathrm{sym}}=D^{-1/2}LD^{-1/2}$\uff0c\u5176\u5143\u7d20\\r\\n$$\\r\\nl_{\\\\mathrm{sym}}(i,j)=\\r\\n\\\\begin{cases}\\r\\n1 & \\\\mathrm{if}\\\\ i=j \\\\\\\\\\r\\n\\\\frac{-1}{\\\\sqrt{d_i d_j}}  & \\\\mathrm{if}\\\\ (v_i,v_j) \\\\in E \\\\\\\\\\r\\n0 & \\\\mathrm{otherwise}\\r\\n\\\\end{cases}\\r\\n$$\\r\\n\\r\\n### \u603b\u53d8\u5dee\\r\\n\\r\\n\u5b9a\u4e49\u5411\u91cf$\\\\boldsymbol{x}=[x_1,x_2,\xb7\xb7\xb7,x_n]^T$\uff0c\u53ef\u8ba4\u4e3a\u662f\u56fe\u4fe1\u53f7\u3002\u5219\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\nL\\\\boldsymbol{x}=(D-A)\\\\boldsymbol{x}&=D\\\\boldsymbol{x} - A \\\\boldsymbol{x}\\\\\\\\\\r\\n&=[\xb7\xb7\xb7, d_ix_i-\\\\sum_{j=1}^{n}a_{ij}x_j,\xb7\xb7\xb7]^T\\r\\n\\\\\\\\\\r\\n&= [\xb7\xb7\xb7,\\\\sum _{j=1}^{n} a_{ij} x_i - \\\\sum _{j=1}^{n} a_{ij} x_j,\xb7\xb7\xb7]^T \\\\\\\\\\r\\n&= [\xb7\xb7\xb7, \\\\sum _{j=1}^{n}a_{ij}(x_i-x_j),\xb7\xb7\xb7]^T\\r\\n\\\\end{aligned}\\r\\n$$\\r\\n\\r\\n\u5206\u91cf $\\\\sum _{j=1}^{n}a_{ij}(x_i-x_j)$ \u53ef\u5199\u6210 $\\\\sum _{j\\\\in N(i)}(x_i-x_j)$\uff0c\u7531\u6b64\u53ef\u77e5\uff0c\u62c9\u666e\u62c9\u65af\u77e9\u9635\u662f\u53cd\u6620\u56fe\u4fe1\u53f7\u5c40\u90e8\u5e73\u6ed1\u5ea6\u7684\u7b97\u5b50\u3002\\r\\n\\r\\n\u63a5\u7740\u6211\u4eec\u5229\u7528\u4e0a\u5f0f\u5b9a\u4e49\u4e8c\u6b21\u578b\uff0c\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\boldsymbol{x}^TL\\\\boldsymbol{x}&=\\\\sum_{i=1}^{n} x_i \\\\sum _{j=1}^{n}a_{ij}(x_i-x_j) \\\\\\\\\\r\\n&= \\\\sum_{i=1}^{n}\\\\sum_{j=1}^{n} a_{ij}(x_i^2-x_ix_j)\\r\\n\\\\end{aligned}\\r\\n$$\\r\\n\u8c03\u6362$i,j$\u7b26\u53f7\uff0c\u6c42\u548c\u987a\u5e8f\u4fdd\u6301\u4e0d\u53d8\uff0c\u6211\u4eec\u5f97\u5230\\r\\n$$\\r\\n\\\\boldsymbol{x}^TL\\\\boldsymbol{x}=\\\\sum_{i=1}^{n}\\\\sum_{j=1}^n a_{ij}(x_i^2-x_ix_j)=\\\\sum_{i=1}^{n}\\\\sum_{j=1}^na_{ij}(x_j^2-x_ix_j)\\r\\n$$\\r\\n\u5c06\u7b49\u5f0f\u5de6\u53f3\u4e24\u8fb9\u76f8\u52a0\uff0c\u4e8e\u662f\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\boldsymbol{x}^TL\\\\boldsymbol{x} &= \\\\frac{1}{2}\\\\sum_{i=1}^{n}\\\\sum_{j=1}^n a_{ij}(x_i^2-2x_ix_j+x_j^2) \\\\\\\\\\r\\n&= \\\\frac{1}{2}\\\\sum_{i=1}^{n}\\\\sum_{j=1}^n a_{ij}(x_i-x_j)^2\\r\\n\\\\end{aligned}\\r\\n$$\\r\\n\\r\\n\u7531\u516c\u5f0f\u53ef\u4ee5\u770b\u51fa\uff0c\u4e8c\u6b21\u578b $\\\\boldsymbol{x}^TL\\\\boldsymbol{x}$ \u80fd\u523b\u753b\u56fe\u4fe1\u53f7\u7684\u603b\u4f53\u5e73\u6ed1\u5ea6\uff0c\u79f0\u4e3a\u603b\u53d8\u5dee\u3002\\r\\n### \u6765\u6e90\\r\\n\\r\\n\u62c9\u666e\u62c9\u65af\u77e9\u9635\u7684\u5b9a\u4e49\u6765\u6e90\u4e8e\u62c9\u666e\u62c9\u65af\u7b97\u5b50\uff0c$n$\u7ef4\u6b27\u5f0f\u7a7a\u95f4\u4e2d\u7684\u4e00\u4e2a\u4e8c\u9636\u5fae\u5206\u7b97\u5b50\uff1a$\\\\Delta f=\\\\sum_{i=1}^n \\\\frac{\\\\partial ^2 f}{\\\\partial x_i^2}$\u3002\u5c06\u8be5\u7b97\u5b50\u9000\u5316\u5230\u79bb\u6563\u4e8c\u7ef4\u56fe\u50cf\u7a7a\u95f4\u5c31\u662f\u8fb9\u7f18\u68c0\u6d4b\u7b97\u5b50\uff1a\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\Delta f(x,y) &=\\r\\n\\\\frac{\\\\partial ^2 f(x,y)}{\\\\partial x^2} + \\r\\n\\\\frac{\\\\partial ^2 f(x,y)}{\\\\partial y^2}\\\\\\\\\\r\\n&= [(f(x+1,y)-f(x,y))-(f(x,y)-f(x-1,y))]\\\\\\\\\\r\\n&+ [(f(x,y+1)-f(x,y))-(f(x,y)-f(x,y-1))]\\\\\\\\\\r\\n&= [f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)] -4f(x,y)\\r\\n\\\\end{aligned}\\r\\n$$\\r\\n\u56fe\u50cf\u5904\u7406\u4e2d\u901a\u5e38\u88ab\u5f53\u4f5c\u6a21\u677f\u7684\u5f62\u5f0f\uff1a\\r\\n$$\\r\\n\\\\begin{bmatrix} 0 & 1 & 0\\\\\\\\ 1 & -4 & 1 \\\\\\\\0 & 1 & 0 \\\\end{bmatrix}\\r\\n$$\\r\\n\u62c9\u666e\u62c9\u65af\u7b97\u5b50\u7528\u6765\u63cf\u8ff0\u4e2d\u5fc3\u50cf\u7d20\u4e0e\u5c40\u90e8\u4e0a\u3001\u4e0b\u3001\u5de6\u3001\u53f3\u56db\u90bb\u5c45\u50cf\u7d20\u7684\u603b\u7684\u5dee\u5f02\uff0c\u8fd9\u79cd\u6027\u8d28\u7ecf\u5e38\u4e5f\u88ab\u7528\u6765\u5f53\u4f5c\u56fe\u50cf\u4e0a\u7684\u8fb9\u7f18\u68c0\u6d4b\u7b97\u5b50\u3002\\r\\n\\r\\n\\r\\n\\r\\n## Laplacian Eigenmaps \\r\\n\\r\\n\u5047\u8bbe\u56fe $G=(V,E)$ \u4e2d\u6709 $n$ \u4e2a\u8282\u70b9\uff0c\u5d4c\u5165\u7ef4\u5ea6\u4e3a $d$\uff0c\u53ef\u5f97\u5230\u5982\u4e0b $n \\\\times d$ \u77e9\u9635 $Y$\uff0c\\r\\n\\r\\n$$\\r\\n\\\\begin{bmatrix}\\r\\ny_1^{(1)} & y_1^{(2)} & \\\\cdots & y_1^{(d)} \\\\\\\\\\r\\ny_2^{(1)} & y_2^{(2)} & \\\\cdots & y_2^{(d)} \\\\\\\\\\r\\n\\\\vdots & \\\\vdots & \\\\ddots& \\\\vdots \\\\\\\\\\r\\ny_n^{(1)} & y_n^{(2)} & \\\\cdots & y_n^{(d)}\\r\\n\\\\end{bmatrix}\\r\\n$$\\r\\n\\r\\n$n$ \u7ef4\u884c\u5411\u91cf $\\\\boldsymbol{y}_k=[y_k^{(1)},y_k^{(2)}, ..., y_k^{(d)}]$\uff0c\u53ef\u8868\u793a\u4e00\u4e2a\u8282\u70b9\u7684Embedding\u3002\\r\\n\\r\\n\u62c9\u666e\u62c9\u65af\u7279\u5f81\u6620\u5c04\uff08Laplacian Eigenmaps\uff09\u7528\u4e8e\u56fe\u5d4c\u5165\u4e2d\uff0c\u5728\u56fe\u4e2d\u90bb\u63a5\u7684\u8282\u70b9\uff0c\u5728\u5d4c\u5165\u7a7a\u95f4\u4e2d\u8ddd\u79bb\u5e94\u8be5\u5c3d\u53ef\u80fd\u7684\u8fd1\u3002\u53ef\u5c06\u5176\u4f5c\u4e3a\u4e00\u9636\u76f8\u4f3c\u6027\u7684\u5b9a\u4e49\uff0c\u56e0\u6b64\u53ef\u4ee5\u5b9a\u4e49\u5982\u4e0b\u7684loss function:\\r\\n\\r\\n$$\\r\\n\\\\mathcal{L}_{1st}=\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij} ||\\\\boldsymbol{y_i}-\\\\boldsymbol{y_j}||_2^2\\r\\n$$\\r\\n\\r\\n\\r\\n\\r\\n$n$ \u7ef4\u5217\u5411\u91cf $\\\\boldsymbol{y}^{(k)} = [y_1^{(k)}, y_2^{(k)}, \xb7\xb7\xb7,y_n^{(k)}]^T$\uff0c\u5bf9\u5e94\u56fe\u4e2d\u6240\u6709\u8282\u70b9\u7684\u7b2c $k$ \u7ef4\u7684\u503c\uff0c\u53ef\u6307\u4e00\u7ec4\u56fe\u4fe1\u53f7\u3002\u56e0\u6b64\u53ef\u4ee5\u5f97\u5230\uff0c\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\mathcal{L}_{1st}=\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij} ||\\\\boldsymbol{y_i}-\\\\boldsymbol{y_j}||_2^2&=\\r\\n\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij} \\\\sum_{k=1}^d (y_{i}^{(k)}-y_{j}^{(k)})^2  \\\\\\\\\\r\\n&= \\\\sum_{k=1}^d \\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij} (y_{i}^{(k)}-y_{j}^{(k)})^2  \\\\\\\\\\r\\n&= 2\\\\sum_{k=1}^d \\\\boldsymbol{y}^{(k)T} L \\\\boldsymbol{y}^{(k)} \\\\\\\\\\r\\n&= 2tr(Y^TLY)\\r\\n\\\\end{aligned}\\r\\n$$\\r\\n\\r\\n$tr(\xb7)$ \u6307\u77e9\u9635\u7684\u8ff9 (trace)\u3002\\r\\n## \u603b\u53d8\u5dee\u7684\u53e6\u4e00\u79cd\u63a8\u5bfc\\r\\n\\r\\n\u5bf9\u4efb\u610f$n$\u7ef4\u5217\u5411\u91cf $\\\\boldsymbol{y}=[y_1, y_2, ..., y_n]^T$\uff0c\u5c55\u5f00\u53ef\u5f97\u5230\uff0c\\r\\n$$\\r\\n\\\\begin{aligned}\\r\\n\\\\boldsymbol{y}^TL\\\\boldsymbol{y}&= \\r\\n\\\\boldsymbol{y}^T D \\\\boldsymbol{y} - \\\\boldsymbol{y}^T A \\\\boldsymbol{y}\\\\\\\\\\r\\n&= \\\\sum_{i=1}^{n} d_iy_i^2- \\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n}  a_{ij} y_iy_j\\\\\\\\\\r\\n&= \\\\frac{1}{2} (\\\\sum_{i=1}^{n} d_iy_i^2  -2\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij}y_iy_j + \\\\sum_{j=1}^{n} d_jy_j^2) \\\\\\\\\\r\\n&=\\\\frac{1}{2} (\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n}  a_{ij}y_i^2  -2\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij}y_iy_j + \\\\sum_{j=1}^{n} \\\\sum_{i=1}^{n}  a_{ji}y_j^2) \\\\\\\\\\r\\n&=\\\\frac{1}{2} (\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n}  a_{ij}y_i^2  -2\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij}y_iy_j + \\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n}  a_{ij}y_j^2) \\\\\\\\\\r\\n&= \\\\frac{1}{2}\\\\sum_{i=1}^{n} \\\\sum_{j=1}^{n} a_{ij}(y_i-y_j)^2\\r\\n\\\\end{aligned}\\r\\n$$\\r\\n\\r\\n\u5176\u4e2d\u5229\u7528\u4e86 $A$ \u662f\u5bf9\u79f0\u77e9\u9635\uff0c$a_{ij}=a_{ji}$"},{"id":"welcome","metadata":{"permalink":"/tiger-website/blog/welcome","source":"@site/blog/2021-08-26-welcome/index.md","title":"Welcome","description":"Docusaurus blogging features are powered by the blog plugin.","date":"2021-08-26T00:00:00.000Z","formattedDate":"August 26, 2021","tags":[{"label":"facebook","permalink":"/tiger-website/blog/tags/facebook"},{"label":"hello","permalink":"/tiger-website/blog/tags/hello"},{"label":"docusaurus","permalink":"/tiger-website/blog/tags/docusaurus"}],"readingTime":0.405,"hasTruncateMarker":false,"authors":[{"name":"S\xe9bastien Lorber","title":"Docusaurus maintainer","url":"https://sebastienlorber.com","imageURL":"https://github.com/slorber.png","key":"slorber"},{"name":"Yangshun Tay","title":"Front End Engineer @ Facebook","url":"https://github.com/yangshun","imageURL":"https://github.com/yangshun.png","key":"yangshun"}],"frontMatter":{"slug":"welcome","title":"Welcome","authors":["slorber","yangshun"],"tags":["facebook","hello","docusaurus"]},"prevItem":{"title":"\u62c9\u666e\u62c9\u65af\u77e9\u9635","permalink":"/tiger-website/blog/\u62c9\u666e\u62c9\u65af\u77e9\u9635"},"nextItem":{"title":"MDX Blog Post","permalink":"/tiger-website/blog/mdx-blog-post"}},"content":"[Docusaurus blogging features](https://docusaurus.io/docs/blog) are powered by the [blog plugin](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-content-blog).\\n\\nSimply add Markdown files (or folders) to the `blog` directory.\\n\\nRegular blog authors can be added to `authors.yml`.\\n\\nThe blog post date can be extracted from filenames, such as:\\n\\n- `2019-05-30-welcome.md`\\n- `2019-05-30-welcome/index.md`\\n\\nA blog post folder can be convenient to co-locate blog post images:\\n\\n![Docusaurus Plushie](./docusaurus-plushie-banner.jpeg)\\n\\nThe blog supports tags as well!\\n\\n**And if you don\'t want a blog**: just delete this directory, and use `blog: false` in your Docusaurus config."},{"id":"mdx-blog-post","metadata":{"permalink":"/tiger-website/blog/mdx-blog-post","source":"@site/blog/2021-08-01-mdx-blog-post.mdx","title":"MDX Blog Post","description":"Blog posts support Docusaurus Markdown features, such as MDX.","date":"2021-08-01T00:00:00.000Z","formattedDate":"August 1, 2021","tags":[{"label":"docusaurus","permalink":"/tiger-website/blog/tags/docusaurus"}],"readingTime":0.175,"hasTruncateMarker":false,"authors":[{"name":"S\xe9bastien Lorber","title":"Docusaurus maintainer","url":"https://sebastienlorber.com","imageURL":"https://github.com/slorber.png","key":"slorber"}],"frontMatter":{"slug":"mdx-blog-post","title":"MDX Blog Post","authors":["slorber"],"tags":["docusaurus"]},"prevItem":{"title":"Welcome","permalink":"/tiger-website/blog/welcome"},"nextItem":{"title":"Long Blog Post","permalink":"/tiger-website/blog/long-blog-post"}},"content":"Blog posts support [Docusaurus Markdown features](https://docusaurus.io/docs/markdown-features), such as [MDX](https://mdxjs.com/).\\n\\n:::tip\\n\\nUse the power of React to create interactive blog posts.\\n\\n```js\\n<button onClick={() => alert(\'button clicked!\')}>Click me!</button>\\n```\\n\\n<button onClick={() => alert(\'button clicked!\')}>Click me!</button>\\n\\n:::"},{"id":"long-blog-post","metadata":{"permalink":"/tiger-website/blog/long-blog-post","source":"@site/blog/2019-05-29-long-blog-post.md","title":"Long Blog Post","description":"This is the summary of a very long blog post,","date":"2019-05-29T00:00:00.000Z","formattedDate":"May 29, 2019","tags":[{"label":"hello","permalink":"/tiger-website/blog/tags/hello"},{"label":"docusaurus","permalink":"/tiger-website/blog/tags/docusaurus"}],"readingTime":2.05,"hasTruncateMarker":true,"authors":[{"name":"Endilie Yacop Sucipto","title":"Maintainer of Docusaurus","url":"https://github.com/endiliey","imageURL":"https://github.com/endiliey.png","key":"endi"}],"frontMatter":{"slug":"long-blog-post","title":"Long Blog Post","authors":"endi","tags":["hello","docusaurus"]},"prevItem":{"title":"MDX Blog Post","permalink":"/tiger-website/blog/mdx-blog-post"},"nextItem":{"title":"First Blog Post","permalink":"/tiger-website/blog/first-blog-post"}},"content":"This is the summary of a very long blog post,\\n\\nUse a `\x3c!--` `truncate` `--\x3e` comment to limit blog post size in the list view.\\n\\n\x3c!--truncate--\x3e\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet\\n\\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet"},{"id":"first-blog-post","metadata":{"permalink":"/tiger-website/blog/first-blog-post","source":"@site/blog/2019-05-28-first-blog-post.md","title":"First Blog Post","description":"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet","date":"2019-05-28T00:00:00.000Z","formattedDate":"May 28, 2019","tags":[{"label":"hola","permalink":"/tiger-website/blog/tags/hola"},{"label":"docusaurus","permalink":"/tiger-website/blog/tags/docusaurus"}],"readingTime":0.12,"hasTruncateMarker":false,"authors":[{"name":"Gao Wei","title":"Docusaurus Core Team","url":"https://github.com/wgao19","image_url":"https://github.com/wgao19.png","imageURL":"https://github.com/wgao19.png"}],"frontMatter":{"slug":"first-blog-post","title":"First Blog Post","authors":{"name":"Gao Wei","title":"Docusaurus Core Team","url":"https://github.com/wgao19","image_url":"https://github.com/wgao19.png","imageURL":"https://github.com/wgao19.png"},"tags":["hola","docusaurus"]},"prevItem":{"title":"Long Blog Post","permalink":"/tiger-website/blog/long-blog-post"}},"content":"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque elementum dignissim ultricies. Fusce rhoncus ipsum tempor eros aliquam consequat. Lorem ipsum dolor sit amet"}]}')}}]);