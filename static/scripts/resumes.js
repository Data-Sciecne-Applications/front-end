$('#btn_submit').click(()=>{
    // Prevent redirection with AJAX for contact form
    var form = $('#resume-form');
    var form_id = 'resume-form';
    var url = form.prop('action');
    var type = form.prop('method');
    var formData = getContactFormData(form_id);
    console.log('formData')
    // submit form via AJAX
    // send_form(form, form_id, url, type, modular_ajax, formData);
});

getContactFormData = (form)=> {
    // creates a FormData object and adds chips text
    var formData = new FormData(document.getElementById(form));
    for (var [key, value] of formData.entries()) { console.log('formData', key, value);}
    return formData
}

send_form = (form, form_id, url, type, inner_ajax, formData) => {

}