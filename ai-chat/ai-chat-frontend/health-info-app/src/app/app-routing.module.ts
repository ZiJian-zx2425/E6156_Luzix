import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ChatComponent } from './chat/chat.component';
import {AuthSuccessComponent} from "./auth-success/auth-success.component";
import {AuthGuard} from "./services/auth/auth.guard";
import {ContactDoctorComponent} from "./contact-doctor/contact-doctor.component"; // Import your ChatComponent

const routes: Routes = [
  { path: '', redirectTo: '/chat', pathMatch: 'full' }, // Default route to ChatComponent
  { path: 'auth-success', component: AuthSuccessComponent },
  { path: 'chat', component: ChatComponent, canActivate: [AuthGuard] },
  { path: 'contact-doctor', component: ContactDoctorComponent,  canActivate: [AuthGuard] }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
