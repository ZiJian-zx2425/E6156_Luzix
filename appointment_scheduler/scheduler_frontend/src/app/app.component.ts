import { Component } from '@angular/core';

@Component({
    selector: 'app-root', // This should match the tag in your index.html
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css'] // You can omit this line if you don't have corresponding CSS
})
export class AppComponent {
    // You can include any logic for the root component here
    title = 'Patient-Doctor Appointment Scheduling'; // This can be used in your app.component.html for dynamic title
}
