import Vue from 'vue'
import Element from 'element-ui'
import locale from 'element-ui/lib/locale/lang/en'
import mathjax from "@/plugin/mathjax";


if (process.env.NODE_ENV !== 'production') {
    import('element-ui/lib/theme-chalk/index.css');
    import('nprogress/nprogress.css');
}

Vue.use(Element, {locale});
Vue.use(mathjax)