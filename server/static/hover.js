const popup = document.createElement('div')
popup.className = 'popup'
popup.style.display = 'none'
document.body.appendChild(popup)

const source_texts = document.querySelectorAll('.src_text')
const source_text_boxes = document.querySelectorAll(".src_text_box")
const timer = clearTimeout(12)

let popupFixated = false

function updatePopup(target) {
        const source_url = target.getAttribute('data-source')
        const chain_text = target.getAttribute('data-chain')
        const source_text = target.getAttribute('data-sourcetext')

        popup.innerHTML = `
        <button id="popup_close">Close</button><br>
        Source URL: <a href="${source_url}">${source_url}</a><br>
        Chain: <pre>${chain_text}</pre><br>
        Matched source text: "${source_text}"
        `
        popup.style.display = 'block'
        popup.style.left = (event.pageX + 10) + 'px'
        popup.style.top = (event.pageY + 10) + 'px'
        popup.querySelector("#popup_close").addEventListener("click", () => {
            popupFixated = false
            popup.style.display = 'none'
        })
}

source_texts.forEach(span => {
    span.addEventListener('mouseenter', (event) => {
        if (popupFixated) {
            return
        }

        updatePopup(event.target)
    })

    span.addEventListener('click', (event) => {
        updatePopup(event.target)
        popupFixated = true
    })
})

source_text_boxes.forEach(box => {
    box.addEventListener('mouseout', event => {
        if (!popupFixated) {
            popup.style.display = 'none'
        }
    })
})

