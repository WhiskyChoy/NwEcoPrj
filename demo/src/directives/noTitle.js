const inserted = (el) => {
    const titleNoe = el.querySelector('title');
    el.removeChild(titleNoe);
    return el;
}

const install = (Vue) => {
    if (install.installed) {
        return
    }
    Vue.directive('noTitle', {inserted});
};

export default install