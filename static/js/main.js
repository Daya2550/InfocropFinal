function googleTranslateElementInit() {
  new google.translate.TranslateElement({pageLanguage: 'en', includedLanguages: 'mr,hi,en', layout: google.translate.TranslateElement.InlineLayout.SIMPLE}, 'google_translate_element');
}

function searchAndScroll() {
  const searchText = document.getElementById("searchInput").value.toLowerCase();
  const headings = document.querySelectorAll("td, h2");

  let found = false;
  for (const heading of headings) {
    const headingText = heading.textContent.toLowerCase();
    if (headingText.includes(searchText)) {
      heading.scrollIntoView({ behavior: "smooth", block: "start" });
      found = true;
      break;
    }
  }
  if (!found) {
    alert("Seaarched Data Not Found.");
  }
}

(function () {
  "use strict";

  /**
   * Easy selector helper function
   */
  const select = (el, all = false) => {
    el = el.trim();
    if (all) {
      return [...document.querySelectorAll(el)];
    } else {
      return document.querySelector(el);
    }
  };

  /**
   * Easy event listener function
   */
  const on = (type, el, listener, all = false) => {
    if (all) {
      select(el, all).forEach((e) => e.addEventListener(type, listener));
    } else {
      select(el, all).addEventListener(type, listener);
    }
  };

  /**
   * Easy on scroll event listener
   */
  const onscroll = (el, listener) => {
    el.addEventListener("scroll", listener);
  };

  /**
   * Sidebar toggle
   */
  if (select(".toggle-sidebar-btn")) {
    on("click", ".toggle-sidebar-btn", function (e) {
      select("body").classList.toggle("toggle-sidebar");
    });
  }

  /**
   * Search bar toggle
   */
  if (select(".search-bar-toggle")) {
    on("click", ".search-bar-toggle", function (e) {
      select(".search-bar").classList.toggle("search-bar-show");
    });
  }

  /**
   * Navbar links active state on scroll
   */
  let navbarlinks = select("#navbar .scrollto", true);
  const navbarlinksActive = () => {
    let position = window.scrollY + 200;
    navbarlinks.forEach((navbarlink) => {
      if (!navbarlink.hash) return;
      let section = select(navbarlink.hash);
      if (!section) return;
      if (
        position >= section.offsetTop &&
        position <= section.offsetTop + section.offsetHeight
      ) {
        navbarlink.classList.add("active");
      } else {
        navbarlink.classList.remove("active");
      }
    });
  };
  window.addEventListener("load", navbarlinksActive);
  onscroll(document, navbarlinksActive);

  /**
   * Toggle .header-scrolled class to #header when page is scrolled
   */
  let selectHeader = select("#header");
  if (selectHeader) {
    const headerScrolled = () => {
      if (window.scrollY > 100) {
        selectHeader.classList.add("header-scrolled");
      } else {
        selectHeader.classList.remove("header-scrolled");
      }
    };
    window.addEventListener("load", headerScrolled);
    onscroll(document, headerScrolled);
  }

  /**
   * Back to top button
   */
  let backtotop = select(".back-to-top");
  if (backtotop) {
    const toggleBacktotop = () => {
      if (window.scrollY > 100) {
        backtotop.classList.add("active");
      } else {
        backtotop.classList.remove("active");
      }
    };
    window.addEventListener("load", toggleBacktotop);
    onscroll(document, toggleBacktotop);
  }

  /**
   * Initiate tooltips
   */
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  /**
   * Initiate quill editors
   */
  if (select(".quill-editor-default")) {
    new Quill(".quill-editor-default", {
      theme: "snow",
    });
  }

  if (select(".quill-editor-bubble")) {
    new Quill(".quill-editor-bubble", {
      theme: "bubble",
    });
  }

  if (select(".quill-editor-full")) {
    new Quill(".quill-editor-full", {
      modules: {
        toolbar: [
          [
            {
              font: [],
            },
            {
              size: [],
            },
          ],
          ["bold", "italic", "underline", "strike"],
          [
            {
              color: [],
            },
            {
              background: [],
            },
          ],
          [
            {
              script: "super",
            },
            {
              script: "sub",
            },
          ],
          [
            {
              list: "ordered",
            },
            {
              list: "bullet",
            },
            {
              indent: "-1",
            },
            {
              indent: "+1",
            },
          ],
          [
            "direction",
            {
              align: [],
            },
          ],
          ["link", "image", "video"],
          ["clean"],
        ],
      },
      theme: "snow",
    });
  }

  /**
   * Initiate TinyMCE Editor
   */
  const useDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const isSmallScreen = window.matchMedia("(max-width: 1023.5px)").matches;

  tinymce.init({
    selector: "textarea.tinymce-editor",
    plugins:
      "preview importcss searchreplace autolink autosave save directionality code visualblocks visualchars fullscreen image link media template codesample table charmap pagebreak nonbreaking anchor insertdatetime advlist lists wordcount help charmap quickbars emoticons",
    editimage_cors_hosts: ["picsum.photos"],
    menubar: "file edit view insert format tools table help",
    toolbar:
      "undo redo | bold italic underline strikethrough | fontfamily fontsize blocks | alignleft aligncenter alignright alignjustify | outdent indent |  numlist bullist | forecolor backcolor removeformat | pagebreak | charmap emoticons | fullscreen  preview save print | insertfile image media template link anchor codesample | ltr rtl",
    toolbar_sticky: true,
    toolbar_sticky_offset: isSmallScreen ? 102 : 108,
    autosave_ask_before_unload: true,
    autosave_interval: "30s",
    autosave_prefix: "{path}{query}-{id}-",
    autosave_restore_when_empty: false,
    autosave_retention: "2m",
    image_advtab: true,
    link_list: [
      {
        title: "My page 1",
        value: "https://www.tiny.cloud",
      },
      {
        title: "My page 2",
        value: "http://www.moxiecode.com",
      },
    ],
    image_list: [
      {
        title: "My page 1",
        value: "https://www.tiny.cloud",
      },
      {
        title: "My page 2",
        value: "http://www.moxiecode.com",
      },
    ],
    image_class_list: [
      {
        title: "None",
        value: "",
      },
      {
        title: "Some class",
        value: "class-name",
      },
    ],
    importcss_append: true,
    file_picker_callback: (callback, value, meta) => {
      /* Provide file and text for the link dialog */
      if (meta.filetype === "file") {
        callback("https://www.google.com/logos/google.jpg", {
          text: "My text",
        });
      }

      /* Provide image and alt text for the image dialog */
      if (meta.filetype === "image") {
        callback("https://www.google.com/logos/google.jpg", {
          alt: "My alt text",
        });
      }

      /* Provide alternative source and posted for the media dialog */
      if (meta.filetype === "media") {
        callback("movie.mp4", {
          source2: "alt.ogg",
          poster: "https://www.google.com/logos/google.jpg",
        });
      }
    },
    templates: [
      {
        title: "New Table",
        description: "creates a new table",
        content:
          '<div class="mceTmpl"><table width="98%%"  border="0" cellspacing="0" cellpadding="0"><tr><th scope="col"> </th><th scope="col"> </th></tr><tr><td> </td><td> </td></tr></table></div>',
      },
      {
        title: "Starting my story",
        description: "A cure for writers block",
        content: "Once upon a time...",
      },
      {
        title: "New list with dates",
        description: "New List with dates",
        content:
          '<div class="mceTmpl"><span class="cdate">cdate</span><br><span class="mdate">mdate</span><h2>My List</h2><ul><li></li><li></li></ul></div>',
      },
    ],
    template_cdate_format: "[Date Created (CDATE): %m/%d/%Y : %H:%M:%S]",
    template_mdate_format: "[Date Modified (MDATE): %m/%d/%Y : %H:%M:%S]",
    height: 600,
    image_caption: true,
    quickbars_selection_toolbar:
      "bold italic | quicklink h2 h3 blockquote quickimage quicktable",
    noneditable_class: "mceNonEditable",
    toolbar_mode: "sliding",
    contextmenu: "link image table",
    skin: useDarkMode ? "oxide-dark" : "oxide",
    content_css: useDarkMode ? "dark" : "default",
    content_style:
      "body { font-family:Helvetica,Arial,sans-serif; font-size:16px }",
  });

  /**
   * Initiate Bootstrap validation check
   */
  var needsValidation = document.querySelectorAll(".needs-validation");

  Array.prototype.slice.call(needsValidation).forEach(function (form) {
    form.addEventListener(
      "submit",
      function (event) {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        }

        form.classList.add("was-validated");
      },
      false
    );
  });

  /**
   * Initiate Datatables
   */
  const datatables = select(".datatable", true);
  datatables.forEach((datatable) => {
    new simpleDatatables.DataTable(datatable, {
      perPageSelect: [5, 10, 15, ["All", -1]],
      columns: [
        {
          select: 2,
          sortSequence: ["desc", "asc"],
        },
        {
          select: 3,
          sortSequence: ["desc"],
        },
        {
          select: 4,
          cellClass: "green",
          headerClass: "red",
        },
      ],
    });
  });

  /**
   * Autoresize echart charts
   */
  const mainContainer = select("#main");
  if (mainContainer) {
    setTimeout(() => {
      new ResizeObserver(function () {
        select(".echart", true).forEach((getEchart) => {
          echarts.getInstanceByDom(getEchart).resize();
        });
      }).observe(mainContainer);
    }, 200);
  }
})();




function goBack() {
  window.history.back();
}

/* =====================================================
   InfoCrop Top Navigation Bar — Auto-injected
   Runs on all crop/plant info pages that load main.js
   ===================================================== */
document.addEventListener('DOMContentLoaded', function () {
  var NAV_H = 52;  // height of ic-topbar

  // ---- 1. Inject font ----
  var fontLink = document.createElement('link');
  fontLink.rel = 'stylesheet';
  fontLink.href = 'https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap';
  document.head.appendChild(fontLink);

  // ---- 2. Inject static CSS (navbar styles only) ----
  var css = document.createElement('style');
  css.id = 'ic-nav-style';
  css.textContent =
    '#ic-topbar{' +
      'position:relative;top:0;left:0;right:0;z-index:99999;' +
      'height:' + NAV_H + 'px;width:100%;' +
      'background:linear-gradient(135deg,#042908,#05380b 60%,#0a5c12);' +
      'display:flex;align-items:center;justify-content:space-between;' +
      'padding:0 20px;box-shadow:0 2px 12px rgba(0,0,0,.4);' +
      'font-family:Outfit,system-ui,sans-serif;flex-shrink:0;' +
    '}' +
    '#ic-topbar a.ic-brand{' +
      'color:#fff;text-decoration:none;font-size:1.05rem;font-weight:700;' +
      'display:flex;align-items:center;gap:8px;white-space:nowrap;flex-shrink:0;' +
    '}' +
    '#ic-topbar ul{' +
      'display:flex;align-items:center;gap:2px;' +
      'list-style:none;margin:0;padding:0;' +
    '}' +
    '#ic-topbar ul a{' +
      'color:rgba(255,255,255,.85);text-decoration:none;' +
      'font-size:.82rem;font-weight:500;padding:5px 9px;' +
      'border-radius:6px;transition:background .18s;white-space:nowrap;display:block;' +
    '}' +
    '#ic-topbar ul a:hover{background:rgba(255,255,255,.18);color:#fff;}' +
    '#ic-topbar a.ic-home-btn{' +
      'color:#fff;text-decoration:none;background:rgba(255,255,255,.15);' +
      'border:1px solid rgba(255,255,255,.3);font-size:.8rem;' +
      'padding:5px 14px;border-radius:18px;transition:background .18s;' +
      'white-space:nowrap;flex-shrink:0;' +
    '}' +
    '#ic-topbar a.ic-home-btn:hover{background:rgba(255,255,255,.25);}' +
    /* Header: FIXED at top:0 — always visible */
    '#header.header,#header.header.fixed-top{' +
      'position:fixed !important;' +
      'top:0 !important;' +
      'left:0 !important;' +
      'right:0 !important;' +
      'width:100% !important;' +
      'height:60px !important;' +
      'z-index:997 !important;' +
    '}' +
    /* Sidebar: fixed, starts right below the fixed header */
    '#sidebar.sidebar{' +
      'position:fixed !important;' +
      'top:60px !important;' +
      'bottom:0 !important;' +
    '}' +
    /* Main: offset down past fixed header only */
    '#main{' +
      'margin-top:60px !important;' +
      'padding:20px 30px !important;' +
    '}' +
    '@media(max-width:820px){#ic-topbar ul{display:none;}}';
  document.head.appendChild(css);

  // ---- 3. Inject HTML ----
  var nav = document.createElement('div');
  nav.id = 'ic-topbar';
  nav.innerHTML =
    '<a href="/" class="ic-brand">&#127807; InfoCrop</a>' +
    '<ul>' +
      '<li><a href="/">&#127968; Home</a></li>' +
      '<li><a href="/home">&#127811; Disease Detection</a></li>' +
      '<li><a href="/CropRec">&#127806; Crop Rec</a></li>' +
      '<li><a href="/PricePrediction">&#128200; Market Price</a></li>' +
      '<li><a href="/WhetherPrediction">&#9925; Weather AI</a></li>' +
      '<li><a href="https://www.apnikheti.com/en/pn/govt-schemes" target="_blank">&#128203; Gov Schemes</a></li>' +
    '</ul>' +
    '<a href="/" class="ic-home-btn">&#8592; Home</a>';

  document.body.insertBefore(nav, document.body.firstChild);
});



