import {Component, OnInit} from '@angular/core';
import {ActivatedRoute, Router} from "@angular/router";
import {AuthService} from "../services/auth/auth.service";

@Component({
  selector: 'app-auth-success',
  templateUrl: './auth-success.component.html',
  styleUrls: ['./auth-success.component.css']
})
export class AuthSuccessComponent implements OnInit{

  constructor(
      private route: ActivatedRoute,
      private authService: AuthService,
      private router: Router
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      const token = params['token'];
      if (token) {
        this.authService.storeToken(token);
        // After storing the token, check the user's role and navigate accordingly
        const userRole = this.authService.getUserRole();
        if (userRole === 'doctor') {
          // If the user is a doctor, navigate to the /officehour route
          this.router.navigate(['/officehour']);
        } else if (userRole === 'patient') {
          // For other roles, navigate to the /schedule route
          this.router.navigate(['/schedule']);
        } else if (userRole === 'volunteer') {
          // this.router.navigate(['/appointments']);
          this.router.navigate(['/appointments']);
        }
      } else {
        console.error('Authentication token is missing');
        this.router.navigate(['/login']);
      }
    });
  }

}
