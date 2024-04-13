// Popup setup:
const popup = document.createElement('div')
popup.className = 'popup'
popup.style.display = 'none'
document.body.appendChild(popup)

// Elements:
const sourceTexts = document.querySelectorAll('.src_text');
const sourceTextBoxes = document.querySelectorAll(".src_text_box");
const loadImagesCheckbox = document.querySelector("#load-images");
const printChainsCheckbox = document.querySelector("#display-chains");

// Parameters:
let popupFixated = false;
let loadImages = loadImagesCheckbox.checked;
let printChains = printChainsCheckbox.checked;


// Functions:
function getKey(elem) {
    const target = elem.closest(".category-table");
    return target.getAttribute('data-key');
}

function updatePopup(event, target) {
    const source_url = target.getAttribute('data-source-url');
    const target_pos = target.getAttribute('data-target-pos');
    const target_likelihood = target.getAttribute('data-target-likelihood');
    let color_class = 'color0';
    target.classList.forEach((cls) => {
        if (cls.startsWith('color')) {
            color_class = cls;
        }
    })

    // INNER HTML SECTION
    popup.innerHTML = `
        <span class="${color_class}">⏺</span>
        <a href="${source_url}" target="_blank" class="source_reference">${source_url}</a><br>
        
        This token: \`<code>${target.innerText}</code>\`
        Matched token: \`<code>${target.getAttribute("data-source-token")}</code>\`
        
        <table>
        <tr><td><b>Target text</b></td><td>
            <span class="greytext">...${target.getAttribute('data-target-text-pre')}</span>${target.getAttribute('data-target-text')}<span class="greytext">${target.getAttribute('data-target-text-post')}...</span>
        </td></tr>
        <tr><td><b>Matched text</b></td><td>
            <span class="greytext">...${target.getAttribute('data-source-text-pre')}</span>${target.getAttribute('data-source-text')}<span class="greytext">${target.getAttribute('data-source-text-post')}...</span>
        </td></tr>    
        </table>
        <b>Score: </b> ${target.getAttribute('data-score')}
        <b>Pos: </b> ${target_pos}
        <b>Likelihood: </b> ${parseFloat(target_likelihood).toFixed(6)}
        <br>`

    if (loadImages) {
        popup.innerHTML += `
        <img alt="plot"
             class="likelihood_plot"
             src="/prev/plots?key=${getKey(target)}&target_pos=${target_pos}&likelihood=${target_likelihood}">
        <br>`
    }

    if (printChains) {
        popup.innerHTML += `<pre>${target.getAttribute('data-chain')}</pre>`
    }

    popup.innerHTML += "<button id=\"popup_close\">Unpin</button>"
    // ! INNER HTML SECTION

    popup.style.display = 'block';
    popup.style.left = (event.pageX + 10) + 'px';
    popup.style.top = (event.pageY - 10 - popup.offsetHeight) + 'px';
    popup.querySelector("#popup_close").addEventListener("click", () => {
        popupFixated = false;
        popup.style.display = 'none';
    })
}

// Add Listeners

loadImagesCheckbox.addEventListener('change', () => {
    loadImages = loadImagesCheckbox.checked;
})

printChainsCheckbox.addEventListener('change', () => {
    printChains = printChainsCheckbox.checked;
})

sourceTexts.forEach(span => {
    span.addEventListener('mousemove', (event) => {
        if (popupFixated) {
            return
        }

        updatePopup(event, event.target)
    })

    span.addEventListener('click', (event) => {
        updatePopup(event, event.target)
        popupFixated = true
    })
})

sourceTextBoxes.forEach(box => {
    box.addEventListener('mouseout', event => {
        if (!popupFixated) {
            popup.style.display = 'none'
        }
    })
})

