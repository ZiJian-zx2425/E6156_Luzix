import {Component, OnInit} from '@angular/core';
import {AuthService} from "./services/auth/auth.service";

@Component({
    selector: 'app-root', // This should match the tag in your index.html
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css'] // You can omit this line if you don't have corresponding CSS
})
export class AppComponent implements OnInit{
    title = 'Patient-Doctor Appointment Scheduling'; // This can be used in your app.component.html for dynamic title
    userRole!: string;

    constructor(private authService: AuthService) {}

    ngOnInit() {
        this.userRole = this.authService.getUserRole();
    }
}
