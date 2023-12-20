import {Component, OnInit} from '@angular/core';
import {AuthService} from "../services/auth/auth.service";

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarComponent implements OnInit {
  userRole: string = '';
  constructor(private authService: AuthService) {}
  ngOnInit(): void {
    // this.userRole = this.authService.getUserRole();
    this.userRole = this.capitalizeFirstLetter(this.authService.getUserRole());
  }

  capitalizeFirstLetter(role: string): string {
    return role.charAt(0).toUpperCase() + role.slice(1);
  }

  onLogout(): void {
    this.authService.logout();
  }
}
